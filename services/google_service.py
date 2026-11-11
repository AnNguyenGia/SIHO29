from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx


class GoogleContextService:
    """Read only the mailbox/calendar belonging to the supplied Google token."""

    def __init__(self, access_token, authenticated_email=None):
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("Không có Google access token; cần kết nối lại.")
        self._access_token = access_token
        self._authenticated_email = authenticated_email

    def _get(self, url, params=None):
        response = httpx.get(url, params=params, headers={
            "Authorization": f"Bearer {self._access_token}"}, timeout=20)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _error(exc):
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 401:
                return "Quyền truy cập đã hết hạn; ngắt kết nối rồi kết nối Google lại."
            if status == 403:
                return "Google chưa cấp đủ quyền hoặc API chưa được bật."
            return f"Google API trả lỗi HTTP {status}."
        return f"Không đọc được dữ liệu Google ({type(exc).__name__})."

    def gmail_context(self, limit=10):
        limit = min(max(int(limit), 1), 10)
        stamp = datetime.now(timezone.utc).isoformat()
        base = "https://gmail.googleapis.com/gmail/v1/users/me"
        result = {"status": "error", "connected": False, "optional": True,
                  "account_email": None, "messages": [], "fetched_at": stamp,
                  "sample_limit": limit, "more_available": False,
                  "scope_note": "Mẫu thư có từ khóa trong 7 ngày; không phải toàn bộ công việc hoặc lịch."}
        try:
            account = self._get(base + "/profile")["emailAddress"]
            # Compare against verified Google identity, never a typed email field.
            if self._authenticated_email and account.casefold() != self._authenticated_email.casefold():
                raise ValueError("Tài khoản Gmail không khớp danh tính Google đã xác thực.")
            query = ('newer_than:7d -in:spam -in:trash '
                     '{deadline assignment exam meeting appointment "hạn chót" "bài tập" "lịch thi"}')
            page = self._get(base + "/messages", {"q": query, "maxResults": limit})
            messages = []
            for item in page.get("messages", [])[:limit]:
                message = self._get(base + "/messages/" + quote(item["id"], safe=""), {
                    "format": "full", "fields": "id,snippet,internalDate,payload/headers"})
                headers = {h["name"].casefold(): h["value"]
                           for h in message.get("payload", {}).get("headers", [])}
                received = datetime.fromtimestamp(int(message["internalDate"]) / 1000, timezone.utc)
                messages.append({"id": message["id"],
                                 "subject": headers.get("subject", "(Không có tiêu đề)")[:300],
                                 "from": headers.get("from", "")[:300],
                                 "received_at": received.isoformat(),
                                 "snippet": message.get("snippet", "")[:1000],
                                 "url": "https://mail.google.com/mail/?authuser=" + quote(account, safe="") + "#search/rfc822msgid%3A"
                                     + quote(headers.get("message-id", ""), safe="") if headers.get("message-id")
                                     else "https://mail.google.com/"})
            result.update(status="ok", connected=True, account_email=account,
                          messages=messages, query=query, more_available=bool(page.get("nextPageToken")))
        except Exception as exc:
            result["error"] = self._error(exc)
        return result

    def calendar_context(self, now, days=3):
        base = "https://www.googleapis.com/calendar/v3"
        result = {"status": "error", "source": "Google Calendar",
                  "fetched_at": datetime.now(timezone.utc).isoformat(),
                  "calendars": [], "events": []}
        try:
            calendars, page_token = [], None
            while True:
                params = {"maxResults": 250}
                if page_token:
                    params["pageToken"] = page_token
                page = self._get(base + "/users/me/calendarList", params)
                calendars.extend(c for c in page.get("items", []) if c.get("primary") or c.get("selected"))
                page_token = page.get("nextPageToken")
                if not page_token:
                    break
            if not calendars:
                raise ValueError("Không có lịch được chọn để đọc.")
            for calendar in calendars:
                result["calendars"].append(calendar.get("summary", calendar["id"]))
                page_token = None
                while True:
                    params = {"timeMin": now.isoformat(), "timeMax": (now + timedelta(days=days)).isoformat(),
                              "singleEvents": "true", "orderBy": "startTime", "maxResults": 250}
                    if page_token:
                        params["pageToken"] = page_token
                    page = self._get(base + "/calendars/" + quote(calendar["id"], safe="") + "/events", params)
                    for item in page.get("items", []):
                        if item.get("status") == "cancelled":
                            continue
                        declined = any(a.get("self") and a.get("responseStatus") == "declined"
                                       for a in item.get("attendees", []))
                        result["events"].append({
                            "title": item.get("summary", "Không có tiêu đề"),
                            "start": item["start"], "end": item["end"],
                            "busy": item.get("transparency") != "transparent" and not declined,
                            "status": item.get("status", "confirmed"),
                            "response_status": "declined" if declined else None,
                            "calendar": calendar.get("summary", calendar["id"]), "url": item.get("htmlLink"),
                        })
                    page_token = page.get("nextPageToken")
                    if len(result["events"]) >= 1000:
                        raise ValueError("Vượt giới hạn 1000 sự kiện; chưa đủ dữ liệu để kết luận lịch trống.")
                    if not page_token:
                        break
            result["status"] = "ok"
        except Exception as exc:
            result["error"] = self._error(exc)
        return result
