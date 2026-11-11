"""Run with: python -m streamlit run app.py (from the SIHO29 project root)."""

import importlib
import logging
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services.context_manager import ContextManager, disconnected_gmail
from services.google_service import GoogleContextService

load_dotenv(Path(__file__).resolve().parent / ".env")
LOGGER = logging.getLogger("siho29")


def redact(value, private_values=()):
    """Remove known credentials without capturing traceback frame locals."""
    text = str(value)
    secrets = [*private_values, *(os.getenv(key, "") for key in (
        "DEEPSEEK_API_KEY", "TAVILY_API_KEY", "GOOGLE_CLIENT_SECRET"))]
    for secret in sorted((v for v in secrets if isinstance(v, str) and v), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)[^\s'\"<>]+", r"\1[REDACTED]", text)
    text = re.sub(r"\b(?:sk-|tvly-|ya29\.)[A-Za-z0-9_.-]+", "[REDACTED]", text)
    return re.sub(r"(?i)([?&](?:access_token|refresh_token|api_key|client_secret)=)[^&\s]+",
                  r"\1[REDACTED]", text)


def error_record(stage, exc, private_values=()):
    details = redact("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), private_values)
    LOGGER.error("SIHO29 | %s\n%s", stage, details)
    return {"stage": stage, "type": type(exc).__name__,
            "message": redact(str(exc) or type(exc).__name__, private_values), "traceback": details}


def attempt(stage, action, issues, private_values=(), on_progress=None):
    if on_progress:
        on_progress(stage)
    try:
        return True, action()
    except Exception as exc:
        issues.append(error_record(stage, exc, private_values))
        return False, None


def render_issue(issue):
    st.error(f"{issue['stage']} — {issue['type']}: {issue['message']}")
    with st.expander(f"Chi tiết lỗi: {issue['stage']}"):
        st.code(issue["traceback"], language="text")


def invalidate_result():
    st.session_state.pop("siho_result", None)


def auth_status():
    """Report configuration problems by key name, never by secret value."""
    problems = []
    try:
        config = st.secrets.get("auth", {})
    except Exception as exc:
        kind = type(exc).__name__
        # Streamlit can wrap a TOML parse failure in SecretNotFoundError too.
        parse_failure = "parsing" in str(exc).lower() or "toml" in kind.lower()
        if not parse_failure and ("NotFound" in kind or isinstance(exc, FileNotFoundError)):
            problems.append("Streamlit chưa tìm thấy secrets.toml. File mẫu secrets.example.toml không được tự đọc.")
        else:
            # Parser exception text can include secret-bearing source lines.
            problems.append(f"Streamlit không đọc được secrets.toml ({kind}). Kiểm tra cú pháp TOML và dấu ngoặc kép.")
        return {"configured": False, "problems": problems}
    if not hasattr(config, "get"):
        return {"configured": False, "problems": ["Cấu hình phải nằm trong bảng [auth] của secrets.toml."]}
    placeholders = ("PASTE_", "THAY_", "CLIENT_", "YOUR_", "<")
    for key in ("client_id", "client_secret", "cookie_secret", "redirect_uri", "server_metadata_url"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"Thiếu hoặc sai kiểu dữ liệu: auth.{key} (cần chuỗi trong dấu ngoặc kép).")
        elif value.strip().upper().startswith(placeholders):
            problems.append(f"auth.{key} vẫn là giá trị mẫu; cần thay bằng giá trị thật.")
    exposed = config.get("expose_tokens", [])
    if not isinstance(exposed, (list, tuple)) or "access" not in exposed:
        problems.append('Thiếu expose_tokens = ["access"] bên trong [auth].')
    client_kwargs = config.get("client_kwargs", {})
    scope = client_kwargs.get("scope", "") if hasattr(client_kwargs, "get") else ""
    scopes = scope.split() if isinstance(scope, str) else scope if isinstance(scope, (list, tuple)) else []
    required_scopes = ("openid", "email", "profile", "https://www.googleapis.com/auth/gmail.readonly",
                       "https://www.googleapis.com/auth/calendar.readonly")
    missing_scopes = [item for item in required_scopes if item not in scopes]
    if missing_scopes:
        problems.append("Thiếu scope trong auth.client_kwargs.scope: " + ", ".join(missing_scopes))
    return {"configured": not problems, "problems": problems}


def auth_configured():
    return auth_status()["configured"]


def google_identity():
    if not auth_configured():
        return None
    user = getattr(st, "user", None)
    if user is None or not getattr(user, "is_logged_in", False):
        return None
    # Never serialize st.user itself: the exposed access token is sensitive.
    return {"id": st.user.get("sub"), "email": st.user.get("email"),
            "verified": st.user.get("email_verified") is True,
            "expires_at": st.user.get("exp", 0)}


def google_access_token(identity):
    if not identity or not identity.get("verified"):
        return None
    try:
        if float(identity.get("expires_at") or 0) <= time.time():
            return None
        tokens = getattr(st.user, "tokens", None)
        token = tokens.get("access") if tokens is not None else None
        return token if isinstance(token, str) and token else None
    except (AttributeError, KeyError, ValueError, TypeError):
        return None


def collect(stage, action, issues, private_values=(), on_progress=None):
    ok, data = attempt(stage, action, issues, private_values, on_progress)
    if not ok:
        return {"status": "unknown", "stage": stage, "error": issues[-1]["message"]}
    if isinstance(data, dict) and data.get("status") in {
        "error", "unknown", "missing", "not_connected", "not_requested"
    }:
        return data
    if data is None or data == "" or data == {}:
        return {"status": "unknown", "error": "Nguồn trả kết quả trống."}
    return {"status": "ok", "data": data, "fetched_at": datetime.now(timezone.utc).isoformat()}


def make_agent(module, name, *args):
    return getattr(importlib.import_module(module), name)(*args)


def validate_decision(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict) or value.get("should") not in {"yes", "no", "uncertain"}:
        raise ValueError("CriticAgent.forward() phải trả dictionary có should=yes/no/uncertain.")
    if not isinstance(value.get("reason"), str):
        raise ValueError("Kết quả critic thiếu reason dạng chuỗi.")
    missing = value.get("missing_information", [])
    if not isinstance(missing, list) or any(not isinstance(item, str) for item in missing):
        raise ValueError("missing_information phải là danh sách chuỗi.")
    for key in ("suggested_plan", "planning_assumptions"):
        items = value.get(key, [])
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f"{key} phải là danh sách chuỗi.")
    if value["should"] == "no":
        value = {**value, "suggested_plan": []}
    return {**value, "missing_information": missing}


def run_check(location, destination="", duration=None, profile=None, token=None,
              identity_email=None, use_gmail=False, use_calendar=False, on_progress=None):
    """Isolate failures and preserve collected evidence, without shared user tokens."""
    issues, private = [], (token or "",)
    profile = {} if profile is None else profile
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "issues": issues,
              "context": {}, "weather": {"status": "unknown"}, "news": {"status": "unknown"},
              "calendar": {"status": "not_connected", "events": [],
                           "error": "Chưa có Calendar của người dùng hiện tại."},
              "gmail": disconnected_gmail()}

    def fallback(reason):
        result["decision"] = {
            "should": "uncertain", "reason": reason,
            "weather": "Xem báo cáo nguồn bên dưới.", "news": "Xem báo cáo nguồn bên dưới.",
            "schedule": "Chưa thể xác nhận lịch và công việc cho kế hoạch này.",
            "missing_information": list(dict.fromkeys([
                *result["context"].get("critical_missing_information", []),
                *(f"{item['stage']}: {item['message']}" for item in issues)])),
        }
        return result

    def validate_inputs():
        if not isinstance(location, str) or not location.strip():
            raise ValueError("Nhập địa chỉ hiện tại, ví dụ Quận 1, TPHCM.")
        if not isinstance(destination, str):
            raise ValueError("Điểm đến phải là chuỗi; có thể để trống khi chưa biết.")
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0):
            raise ValueError("Thời lượng phải là số phút lớn hơn 0.")
        if not isinstance(profile, dict):
            raise ValueError("Hồ sơ người dùng phải là dictionary.")

    ok, _ = attempt("Kiểm tra dữ liệu nhập", validate_inputs, issues, private, on_progress)
    if not ok:
        return fallback("Cần sửa dữ liệu nhập trước khi kiểm tra.")
    location, destination = location.strip(), destination.strip()
    ok, critic = attempt("Khởi tạo agent đánh giá",
                         lambda: make_agent("agents.critic_agent", "CriticAgent", location),
                         issues, private, on_progress)
    if not ok:
        return fallback("Chưa khởi tạo được bước đánh giá. Xem nguyên nhân cụ thể bên dưới.")

    def interpret_addresses():
        value = critic.analyze_addresses(destination or None)
        if not isinstance(value, dict):
            raise ValueError("analyze_addresses() phải trả dictionary.")
        for prefix in ("origin", "destination"):
            if value.get(prefix + "_scope") not in {"vietnam", "outside_vietnam", "unknown"}:
                raise ValueError("Agent phân tích địa chỉ trả phạm vi không hợp lệ.")
            city = value.get(prefix + "_city")
            if city is not None and (not isinstance(city, str) or not city.strip()):
                raise ValueError("Tên thành phố do agent suy luận phải là chuỗi hoặc null.")
        return value

    ok, interpretation = attempt("Agent phân tích địa chỉ", interpret_addresses, issues, private, on_progress)
    if not ok:
        interpretation = {"origin_city": None, "origin_scope": "unknown", "destination_city": None,
                          "destination_scope": "unknown", "note": "Chưa phân tích được địa chỉ; tiếp tục thử các nguồn với địa chỉ gốc."}
    result["address_interpretation"] = interpretation
    if interpretation["origin_scope"] == "outside_vietnam" or (
            destination and interpretation["destination_scope"] == "outside_vietnam"):
        return fallback("Địa chỉ được agent nhận diện ở ngoài Việt Nam. SIHO29 hiện chỉ hỗ trợ Việt Nam.")
    # The raw address is preserved for news/context. A city inferred by the LLM
    # is only a city-level weather hint, never a verified street coordinate.
    origin_query = f"{interpretation['origin_city']}, Vietnam" if interpretation.get("origin_city") else location
    destination_query = (f"{interpretation['destination_city']}, Vietnam"
                         if interpretation.get("destination_city") and destination else destination)
    ok, now = attempt("Đọc múi giờ từ CriticAgent", lambda: datetime.now(critic.timezone),
                      issues, private, on_progress)
    if not ok:
        return fallback("CriticAgent cần thuộc tính timezone là ZoneInfo của địa điểm.")
    result["checked_at"] = now.isoformat()

    def weather_at(place):
        return make_agent("agents.weather_agent", "WeatherAgent").forward(place)

    origin_weather = collect("Thời tiết tại vị trí hiện tại", lambda: weather_at(origin_query),
                             issues, private, on_progress)
    target_weather = (origin_weather if not destination else
        origin_weather if destination_query == origin_query else
        collect("Thời tiết tại điểm đến", lambda: weather_at(destination_query), issues, private, on_progress))
    result["weather"] = {
        "status": "ok" if origin_weather["status"] == target_weather["status"] == "ok" else "unknown",
        "origin": origin_weather, "destination": target_weather,
        "destination_scope": "specified_destination_city" if destination else "origin_area_only"}
    news_query = f"{destination or location}, Việt Nam"
    result["news"] = collect("Tin tức tại điểm đến hoặc quanh địa chỉ hiện tại",
        lambda: make_agent("agents.new_agent", "NewAgent").forward(news_query), issues, private, on_progress)

    if token and identity_email:
        ok, service = attempt("Khởi tạo kết nối Google", lambda: GoogleContextService(token, identity_email),
                               issues, private, on_progress)
        if ok:
            result["calendar"] = {"status": "not_requested", "events": [], "error": "Chưa bật dùng Calendar."}
            result["gmail"].update(status="not_requested", connected=True, account_email=identity_email)
            for enabled, name, label, action in (
                (use_calendar, "calendar", "Đọc Google Calendar", lambda: service.calendar_context(now)),
                (use_gmail, "gmail", "Đọc Gmail đã được cho phép", lambda: service.gmail_context(limit=10)),
            ):
                if enabled:
                    ok, data = attempt(label, action, issues, private, on_progress)
                    if ok and isinstance(data, dict):
                        result[name] = data
                    else:
                        result[name] = {"status": "error", "error": "Không lấy được dữ liệu nguồn."}

    def build_context():
        return ContextManager().build(
            current_location=location, timezone=critic.timezone.key, now=now,
            destination=destination or None, duration_minutes=duration, profile=profile,
            gmail_context=result["gmail"], calendar_context=result["calendar"], automatic_planning=True)

    ok, context = attempt("Tạo context cho người dùng", build_context, issues, private, on_progress)
    if not ok:
        return fallback("Không tổng hợp được context; các báo cáo đã lấy vẫn được giữ lại.")
    result["context"] = context
    context["address_interpretation"] = interpretation
    context["address_evidence_note"] = "Địa chỉ được AI diễn giải; chưa xác minh tọa độ, tuyến đường hoặc địa chỉ đường phố."

    def evaluate():
        decision = validate_decision(critic.forward(
            weather=result["weather"], schedule=result["calendar"], news=result["news"], user_context=context))
        missing = [f"Nguồn {name} chưa có dữ liệu đầy đủ." for name in ("weather", "calendar", "news")
                   if result[name].get("status") != "ok"]
        if interpretation["origin_scope"] == "unknown":
            missing.append("Chưa xác định được khu vực của địa chỉ hiện tại.")
        if destination and interpretation["destination_scope"] == "unknown":
            missing.append("Chưa xác định được khu vực của địa chỉ điểm đến.")
        return ContextManager.apply_guard(decision, {**context, "critical_missing_information": [
            *context.get("critical_missing_information", []), *missing]})

    ok, decision = attempt("CriticAgent đánh giá và trả kết quả", evaluate, issues, private, on_progress)
    if not ok:
        return fallback("Bước đánh giá thất bại. Chưa thể kết luận nên đi chơi; xem báo cáo nguồn và lỗi bên dưới.")
    result["decision"] = decision
    return result


def main():
    st.set_page_config(page_title="SIHO29", page_icon="🌙", layout="wide")
    identity = google_identity()
    actor = (identity.get("id") or identity.get("email")) if identity else "anonymous"
    if st.session_state.get("siho_actor") != actor:
        for key in list(st.session_state):
            if key.startswith("siho_"):
                del st.session_state[key]
        st.session_state["siho_actor"] = actor

    st.title("SIHO29 · Tối nay có nên đi chơi?")
    st.caption("Đối chiếu thời tiết, lịch, tin tức và bối cảnh cá nhân tại Việt Nam. Kết nối Google là tùy chọn.")

    with st.sidebar:
        st.subheader("Google / Gmail · tùy chọn")
        email_input = st.text_input("Địa chỉ Gmail hoặc Google Workspace", value=identity["email"] if identity else "",
                                   disabled=bool(identity), key="siho_email")
        st.caption("Nhập email không cấp quyền. Trên Google, hãy chọn tài khoản muốn kết nối; tài khoản thực tế sẽ hiện sau đăng nhập.")
        oauth_status = auth_status()
        configured = oauth_status["configured"]
        if not identity:
            if st.button("Kết nối Google", disabled=not configured):
                if email_input and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email_input.strip()):
                    st.error("Địa chỉ email chưa hợp lệ.")
                else:
                    try:
                        st.login()
                    except Exception as exc:
                        config = st.secrets.get("auth", {})
                        render_issue(error_record("Đăng nhập Google", exc, (
                            config.get("client_secret", ""), config.get("cookie_secret", ""))))
            if not configured:
                st.info("Kết nối Google chưa sẵn sàng. Vẫn có thể Check mà không kết nối Google.")
                with st.expander("Kiểm tra cấu hình Google", expanded=True):
                    for problem in oauth_status["problems"]:
                        st.warning(problem)
                    expected_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
                    st.caption("Đường dẫn cấu hình của project:")
                    st.code(str(expected_path), language="text")
                    if Path.cwd().resolve() != Path(__file__).resolve().parent:
                        st.warning("Bạn đang chạy từ thư mục khác project. Hãy cd vào thư mục chứa app.py rồi chạy lại Streamlit.")
                    st.caption("Tên đúng là secrets.toml, nằm trong .streamlit/. Sau khi lưu, dừng và khởi động lại Streamlit. Không gửi client_secret hoặc cookie_secret vào chat.")
        else:
            st.success(f"Tài khoản đã đăng nhập: {identity['email']}")
            if st.button("Ngắt kết nối Google"):
                invalidate_result()
                st.logout()

        token = google_access_token(identity)
        if identity and not token:
            st.warning("Không có access token hợp lệ. Kiểm tra cấu hình expose_tokens hoặc kết nối Google lại.")
        use_calendar = st.checkbox("Dùng Calendar của tài khoản này", value=bool(token),
                                   disabled=not bool(token), key="siho_use_calendar", on_change=invalidate_result)
        use_gmail = st.checkbox("Dùng email gần đây làm context cho AI", value=False,
                                disabled=not bool(token), key="siho_use_gmail", on_change=invalidate_result)
        st.caption("Khi bật email và bấm Check: đọc tối đa 10 thư có từ khóa trong 7 ngày, chỉ đưa tiêu đề, người gửi và trích đoạn vào AI. Không gửi, sửa hoặc xóa thư.")

    location = st.text_input(
        "Địa chỉ hiện tại của bạn", placeholder="Quận 1, TPHCM hoặc số nhà, đường, phường, thành phố",
        key="siho_location", on_change=invalidate_result)
    destination = st.text_input(
        "Địa chỉ nơi muốn đến (tùy chọn)", placeholder="Để trống để gợi ý quanh khu vực hiện tại",
        key="siho_destination", on_change=invalidate_result)
    st.caption("Nhập địa chỉ tự do trong Việt Nam. Agent tự phân tích khu vực và đề xuất kế hoạch tối nay từ dữ liệu hiện có.")

    if st.button("Check", type="primary", use_container_width=True):
        invalidate_result()
        progress = st.empty()
        if not location.strip():
            st.error("Hãy nhập địa chỉ hiện tại của bạn.")
        else:
            try:
                with st.spinner("Đang tổng hợp context và kiểm tra các nguồn dữ liệu…"):
                    st.session_state["siho_result"] = run_check(
                        location.strip(), destination.strip(), profile={},
                        token=token, identity_email=identity["email"] if identity else None,
                        use_gmail=use_gmail, use_calendar=use_calendar,
                        on_progress=lambda stage: progress.info(f"Đang thực hiện: {stage}"))
            except Exception as exc:
                render_issue(error_record("Điều phối nút Check", exc, (token or "",)))
            finally:
                progress.empty()

    result = st.session_state.get("siho_result")
    if result:
        decision = result["decision"]
        labels = {"yes": "Có thể cân nhắc đi", "no": "Nên hoãn hoặc đổi kế hoạch", "uncertain": "Chưa đủ thông tin"}
        st.subheader(labels.get(decision["should"], decision["should"]))
        st.write(decision["reason"])
        for issue in result.get("issues", []):
            render_issue(issue)
        for column, name, label in zip(st.columns(3), ("weather", "schedule", "news"), ("Thời tiết", "Lịch và công việc", "Tin tức")):
            with column:
                st.markdown(f"**{label}**")
                st.write(decision.get(name, "Chưa có nhận xét."))
        if decision.get("suggested_plan"):
            st.markdown("**Kế hoạch gợi ý tối nay**")
            st.caption("Các giờ hoặc thời lượng do agent đề xuất là ước tính, không phải lịch đã xác nhận.")
            for item in decision["suggested_plan"]:
                st.write(f"• {item}")
        if decision.get("planning_assumptions"):
            st.info("Các giả định của kế hoạch: " + "; ".join(decision["planning_assumptions"]))
        if decision.get("missing_information"):
            st.warning("\n".join(decision["missing_information"]))
        for name, label in (("weather", "Thời tiết"), ("calendar", "Calendar"),
                            ("news", "Tin tức"), ("gmail", "Gmail")):
            report = result[name]
            if report.get("status") in {"error", "unknown"} and report.get("error"):
                st.warning(f"{label}: {report['error']}")
        with st.expander("Context đã dùng cho lần Check này"):
            st.json(result["context"])
        with st.expander("Báo cáo nguồn đã thu thập"):
            st.json({name: result[name] for name in ("calendar", "weather", "news", "gmail")})
        st.caption(f"Kiểm tra lúc {result['checked_at']}. Email là nguồn bổ sung, không xác nhận mọi deadline hay công việc đã hoàn thành.")


if __name__ == "__main__":
    main()
