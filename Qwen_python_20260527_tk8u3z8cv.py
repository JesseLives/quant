#!/usr/bin/env python3
import argparse
import ipaddress
import socket
import sys
import json
import threading
import time
import re
import signal
import base64
import ssl
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

# --- Suppress SSL warnings for self-signed camera certs ---
def _try_import(name):
    try:
        mod = __import__(name)
        if name == "requests":
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return mod, True
    except ImportError:
        return None, False

requests, _HAS_REQUESTS = _try_import("requests")
cv2, _HAS_CV2 = _try_import("cv2")

# --- Thread-safe print ---
_print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)

# --- Configuration ---
CAMERA_PORTS = {
    80:    "HTTP (Web)",
    443:   "HTTPS (Web)",
    554:   "RTSP (Video)",
    8080:  "HTTP Alt",
    8000:  "Hikvision HTTP",
    37777: "Dahua TCP",
    37778: "Dahua UDP",
    9000:  "Hikvision Alt",
    82:    "HTTP Alt",
    81:    "HTTP Alt",
    5000:  "HTTP Alt / Synology",
    1025:  "RTSP Alt",
    1935:  "RTMP (Stream)",
    49152: "UPnP/IGD",
}

DEFAULT_CREDENTIALS = {
    "generic": [
        ("admin", "admin"), ("admin", "12345"), ("admin", "123456"),
        ("admin", ""), ("admin", "password"), ("root", "root"),
        ("root", "admin"), ("root", "camera"), ("root", "pass"),
        ("root", "12345"), ("Administrator", "admin"),
        ("admin", "1111"), ("admin", "9999"),
    ],
    "hikvision": [
        ("admin", "12345"), ("admin", ""), ("admin", "hikvision"),
        ("admin", "12345abc"),
    ],
    "dahua": [
        ("admin", "admin"), ("admin", ""),
        ("666666", "666666"), ("888888", "888888"),
    ],
    "axis": [("root", "pass"), ("root", "admin")],
    "foscam": [("admin", ""), ("admin", "admin")],
    "dlink": [("admin", ""), ("admin", "admin")],
    "tp_link": [("admin", "admin")],
    "reolink": [("admin", ""), ("admin", "admin")],
    "wyze": [("admin", "12345")],
    "vstarcam": [("admin", "admin")],
    "xiongmai": [("admin", ""), ("admin", "admin")],
    "hanwha": [("admin", "4321"), ("admin", "1234")],
    "ubiquiti": [("ubnt", "ubnt"), ("admin", "admin")],
}

BRAND_ENDPOINTS = {
    "hikvision": [
        "/ISAPI/System/deviceInfo",
        "/ISAPI/Security/userCheck",
        "/SDK/webLanguage",
    ],
    "dahua": [
        "/RPC2_Login",
        "/cgi-bin/magicBox.cgi?action=getManufacturer",
        "/cgi-bin/magicBox.cgi?action=getDeviceType",
    ],
    "axis": [
        "/axis-cgi/admin/param.cgi?action=list",
        "/axis-cgi/param.cgi?action=list",
        "/vapix/param.cgi?action=list",
    ],
    "foscam": [
        "/cgi-bin/CGIProxy.fcgi?cmd=getDevState&usr=admin&pwd=",
        "/cgi-bin/CGIProxy.fcgi?cmd=getPtzSpeed&usr=admin&pwd=",
    ],
    "dlink": [
        "/config/getuser?index=0",
        "/dms?nowprofileid",
    ],
    "generic": [
        "/", "/index.html", "/login.html",
        "/doc/page/login.asp", "/web/index.html",
        "/viewer/live/index.html",
    ],
}

RTSP_PATTERNS = {
    "hikvision": "rtsp://{user}:{pw}@{ip}:554/Streaming/Channels/101",
    "dahua": "rtsp://{user}:{pw}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    "axis": "rtsp://{user}:{pw}@{ip}/axis-media/media.amp",
    "foscam": "rtsp://{user}:{pw}@{ip}:554/videoMain",
    "generic": "rtsp://{user}:{pw}@{ip}:554/live/ch0",
}

VERIFY_ENDPOINTS = [
    "/ISAPI/System/deviceInfo",
    "/ISAPI/ContentMgmt/record/capabilities",
    "/ISAPI/Security/adminAccount",
    "/cgi-bin/magicBox.cgi?action=getDeviceType",
    "/cgi-bin/configManager.cgi?action=getConfig&name=General",
    "/axis-cgi/param.cgi?action=list",
    "/onvif/device_service",
]

BRAND_KEYWORDS = {
    "hikvision": ["hikvision", "hik-web", "hik-connect", "hik-central"],
    "dahua": ["dahua", "dmss", "dvr"],
    "axis": ["axis communications", "axis camera", "axis web server"],
    "foscam": ["foscam"],
    "dlink": ["d-link", "dlink"],
    "reolink": ["reolink"],
    "vstarcam": ["vstarcam"],
    "tp_link": ["tp-link", "tplink"],
    "xiongmai": ["xiongmai", "xmeye"],
    "hanwha": ["hanwha", "techwin", "wisenet"],
    "ubiquiti": ["ubiquiti", "uvc"],
}

ONVIF_SOAP_BODY = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
  <s:Body xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xmlns:xsd="http://www.w3.org/2001/XMLSchema">
    <GetDeviceInformation xmlns="http://www.onvif.org/ver10/device/wsdl"/>
  </s:Body>
</s:Envelope>"""
ONVIF_SOAP_ACTION = '"http://www.onvif.org/ver10/device/wsdl/GetDeviceInformation"'

# --- Enhanced Rate Limiter (Token Bucket) ---
class TokenBucketRateLimiter:
    """Token bucket rate limiter for bursty network behavior."""
    def __init__(self, rate_per_sec, burst_size=None):
        self.rate = rate_per_sec
        self.burst_size = burst_size or max(1, int(rate_per_sec))
        self.lock = threading.Lock()
        self.last_time = time.monotonic()
        self.tokens = float(self.burst_size)

    def acquire(self) -> bool:
        if self.rate <= 0:
            return True  # No limit
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_time
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            self.last_time = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            
            # Calculate exact wait time needed for 1 token
            wait_time = (1.0 - self.tokens) / self.rate
        
        time.sleep(max(0.001, wait_time))
        
        # Recalculate tokens after exact sleep
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_time
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            self.last_time = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            # Edge case: floating-point rounding may leave tokens < 1.0
            # Always consume one token after sleeping to prevent limit bypass
            self.tokens = max(0.0, self.tokens - 1.0)
            return True

# --- Port scan result with error info ---
@dataclass
class PortScanResult:
    port: int
    is_open: bool
    error: Optional[str] = None

# --- Scan result ---
@dataclass
class ScanResult:
    ip: str
    open_ports: list = field(default_factory=list)
    port_scan_details: list = field(default_factory=list)
    detected_brand: str = "unknown"
    brand_source: str = ""           # "header" / "content" / "onvif" / "none"
    login_page_detected: bool = False
    default_login_found: tuple = None
    auth_type_used: str = "none"
    credentials_verified: bool = False
    default_config_accessible: bool = False
    config_details: dict = field(default_factory=dict)
    rtsp_accessible: bool = False
    http_banner: str = ""
    errors: list = field(default_factory=list)
    scan_duration: float = 0.0

class CCTVScanner:
    def __init__(
        self,
        network=None,
        target=None,
        timeout=3,
        threads=30,
        cred_threads=6,
        quick=False,
        verify_creds=True,
        check_rtsp=True,
        use_onvif=True,
        rate_limit=0,
        creds_file=None,
    ):
        self.timeout = timeout
        self.threads = threads
        self.cred_threads = cred_threads
        self.quick = quick
        self.verify_creds = verify_creds
        self.check_rtsp = check_rtsp
        self.use_onvif = use_onvif
        self.rate_limiter = TokenBucketRateLimiter(rate_limit) if rate_limit > 0 else None
        self.creds_file = creds_file
        self.results = []
        self.results_lock = threading.Lock()
        self.scanned = 0
        self._local = threading.local()  # Thread-local session storage
        self._executor: Optional[ThreadPoolExecutor] = None
        self.custom_credentials = self._load_custom_credentials()

        if network:
            # Use iterator to avoid OOM on large networks - process targets lazily
            self.targets = list(ipaddress.IPv4Network(network, strict=False).hosts())[:1024]  # Cap at 1024 hosts for safety
            if len(self.targets) >= 1024:
                safe_print(f"[!] Warning: Target list capped to first 1024 hosts. Full network has {len(list(ipaddress.IPv4Network(network, strict=False).hosts()))} hosts.")
        elif target:
            self.targets = [ipaddress.IPv4Address(target)]
        else:
            # This should never happen due to CLI argument requirements
            raise ValueError("Either --network or --target must be specified")

    def _load_custom_credentials(self) -> Dict[str, List[Tuple[str, str]]]:
        if not self.creds_file:
            return {}
        try:
            with open(self.creds_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for brand, creds in data.items():
                    if not isinstance(creds, list):
                        safe_print(f"[!] Invalid credentials format for brand '{brand}'")
                        continue
                    for cred in creds:
                        if not (isinstance(cred, (list, tuple)) and len(cred) == 2):
                            safe_print(f"[!] Invalid credential format: {cred}")
                return data
        except FileNotFoundError:
            safe_print(f"[!] Credentials file not found: {self.creds_file}")
        except json.JSONDecodeError:
            safe_print(f"[!] Invalid JSON in credentials file: {self.creds_file}")
        return {}

    def _get_credentials_for_brand(self, brand: str) -> List[Tuple[str, str]]:
        creds = []
        if brand in self.custom_credentials:
            creds.extend(self.custom_credentials[brand])
        if brand in DEFAULT_CREDENTIALS:
            creds.extend(DEFAULT_CREDENTIALS[brand])
        if brand != "generic":
            if "generic" in self.custom_credentials:
                creds.extend(self.custom_credentials["generic"])
            creds.extend(DEFAULT_CREDENTIALS["generic"])
            
        seen = set()
        unique_creds = []
        for u, p in creds:
            if (u, p) not in seen:
                seen.add((u, p))
                unique_creds.append((u, p))
        return unique_creds

    def _progress(self):
        with self.results_lock:
            self.scanned += 1
            return self.scanned

    def _build_url(self, ip, port, path="/"):
        scheme = "https" if port == 443 else "http"
        path = path.lstrip("/")
        return f"{scheme}://{ip}:{port}/{path}" if path else f"{scheme}://{ip}:{port}/"

    # ------------------------------------------------------------------------
    #  HTTP / Auth helpers with thread-local session reuse
    # ------------------------------------------------------------------------
    def _get_session(self):
        if not hasattr(self._local, 'session'):
            if _HAS_REQUESTS:
                self._local.session = requests.Session()
                self._local.session.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                })
            else:
                self._local.session = None
        return self._local.session

    def _fetch(self, url, username=None, password=None, method="GET", timeout=None, session=None):
        t = timeout or self.timeout
        sess = session or self._get_session()
        if sess is not None:
            return self._fetch_requests(sess, url, username, password, method, t)
        else:
            return self._fetch_urllib(url, username, password, method, t)

    def _fetch_requests(self, session, url, username, password, method, timeout):
        from requests.auth import HTTPBasicAuth, HTTPDigestAuth
        try:
            if username is not None:
                resp = session.request(
                    method, url, auth=HTTPBasicAuth(username, password or ""),
                    timeout=timeout, allow_redirects=True, verify=False,
                )
                if resp.status_code == 401:
                    www_auth = resp.headers.get("WWW-Authenticate", "").lower()
                    if "digest" in www_auth:
                        resp = session.request(
                            method, url, auth=HTTPDigestAuth(username, password or ""),
                            timeout=timeout, allow_redirects=True, verify=False,
                        )
                return resp.status_code, resp.text, dict(resp.headers)
            else:
                resp = session.request(
                    method, url, timeout=timeout, allow_redirects=True, verify=False,
                )
                return resp.status_code, resp.text, dict(resp.headers)
        except (requests.RequestException, ValueError, OSError, ssl.SSLError) as e:
            return None, "", {"_error": str(e)}
        except Exception as e:
            return None, "", {"_error": f"Unexpected: {e}"}

    def _fetch_urllib(self, url, username, password, method, timeout):
        """Urllib fallback with native Digest/Basic Auth handling."""
        try:
            req = urllib.request.Request(url, method=method)
            req.add_header("User-Agent", "Mozilla/5.0")
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            handlers = [urllib.request.HTTPSHandler(context=ctx)]
            
            if username is not None:
                # Native standard library handling for Basic & Digest (RFC 2617 compliant)
                password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
                password_mgr.add_password(None, url, username, password or "")
                handlers.append(urllib.request.HTTPBasicAuthHandler(password_mgr))
                handlers.append(urllib.request.HTTPDigestAuthHandler(password_mgr))
                
            opener = urllib.request.build_opener(*handlers)
            resp = opener.open(req, timeout=timeout)
            content = resp.read(8192).decode("utf-8", errors="ignore")
            return resp.status, content, dict(resp.headers)
            
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read(8192).decode("utf-8", errors="ignore")
            except Exception:
                pass
            return e.code, body, dict(e.headers) if hasattr(e, "headers") else {}
            
        except (urllib.error.URLError, socket.timeout, ssl.SSLError, ConnectionError) as e:
            return None, "", {"_error": str(e)}
        except Exception as e:
            return None, "", {"_error": f"Unexpected: {e}"}

    # ------------------------------------------------------------------------
    #  Enhanced port scanning
    # ------------------------------------------------------------------------
    def _check_port(self, ip, port) -> PortScanResult:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            result = s.connect_ex((str(ip), port))
            s.close()
            if result == 0:
                return PortScanResult(port=port, is_open=True)
            else:
                return PortScanResult(port=port, is_open=False, error="Connection refused")
        except socket.timeout:
            return PortScanResult(port=port, is_open=False, error="Timeout")
        except socket.error as e:
            return PortScanResult(port=port, is_open=False, error=str(e))
        except Exception as e:
            return PortScanResult(port=port, is_open=False, error=f"Unexpected: {e}")

    def _scan_ports_parallel(self, ip, ports):
        results = []
        # Reduce inner thread pool size to avoid thread explosion (30 hosts * 20 workers = 600 threads)
        with ThreadPoolExecutor(max_workers=min(len(ports), 5)) as ex:
            futs = {ex.submit(self._check_port, ip, p): p for p in ports}
            for f in as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append(PortScanResult(port=futs[f], is_open=False, error=str(e)))
        return results

    # ------------------------------------------------------------------------
    #  Brand detection
    # ------------------------------------------------------------------------
    def _detect_brand(self, ip, port):
        url = self._build_url(ip, port)
        status, content, hdrs = self._fetch(url, method="HEAD")
        if hdrs and not hdrs.get("_error"):
            server = hdrs.get("Server", hdrs.get("server", "")).lower()
            if server:
                for brand, kws in BRAND_KEYWORDS.items():
                    if any(kw in server for kw in kws):
                        return brand, "header"
                        
        status, content, hdrs = self._fetch(url)
        if content and not hdrs.get("_error"):
            cl = content.lower()
            for brand, kws in BRAND_KEYWORDS.items():
                if any(kw in cl for kw in kws):
                    return brand, "content"
                    
        if self.use_onvif and port in (80, 443, 8000, 8080, 554):
            onvif_brand = self._detect_brand_onvif(ip, port)
            if onvif_brand != "unknown":
                return onvif_brand, "onvif"
                
        return "unknown", "none"

    def _detect_brand_onvif(self, ip, port):
        soap_headers = {
            "Content-Type": "application/soap+xml; charset=utf-8",
            "User-Agent": "Mozilla/5.0",
            "SOAPAction": ONVIF_SOAP_ACTION,
        }
        
        if port == 443:
            schemes = ["https"]
        elif port == 554:
            schemes = ["http", "https"]
        else:
            schemes = ["http", "https"]
            
        endpoints = ["/onvif/device_service", "/onvif/services"]
        
        for scheme in schemes:
            for ep in endpoints:
                url = f"{scheme}://{ip}:{port}{ep}"
                try:
                    session = self._get_session()
                    if session is None:
                        req = urllib.request.Request(
                            url, data=ONVIF_SOAP_BODY.encode("utf-8"),
                            headers=soap_headers, method="POST"
                        )
                        ctx = ssl.create_default_context()
                        ctx.check_hostname = False
                        ctx.verify_mode = ssl.CERT_NONE
                        opener = urllib.request.build_opener(
                            urllib.request.HTTPSHandler(context=ctx)
                        )
                        resp = opener.open(req, timeout=2)
                        text = resp.read(4096).decode("utf-8", errors="ignore")
                    else:
                        resp = session.post(
                            url, data=ONVIF_SOAP_BODY.encode("utf-8"),
                            headers=soap_headers, timeout=2, verify=False
                        )
                        text = resp.text
                        
                    if "Manufacturer" in text:
                        match = re.search(
                            r'<(?:[^>]*:)?Manufacturer>(.*?)</(?:[^>]*:)?Manufacturer>',
                            text,
                            re.IGNORECASE | re.DOTALL
                        )
                        if match:
                            manu = match.group(1).lower()
                            for brand, kws in BRAND_KEYWORDS.items():
                                if any(kw in manu for kw in kws):
                                    return brand
                except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, ssl.SSLError, ConnectionError, OSError):
                    continue
                except Exception as e:
                    # Gracefully handle requests not being available (_HAS_REQUESTS is False)
                    if _HAS_REQUESTS:
                        try:
                            if isinstance(e, requests.RequestException):
                                continue
                        except (AttributeError, TypeError):
                            pass
                    continue
        return "unknown"

    def _is_login_page(self, code, content):
        if code in (401, 403):
            return True
        if not content:
            return False
        cl = content.lower()
        indicators = [
            "login", "sign in", "password", "username", "authentication",
            "camera", "surveillance", "nvr", "dvr",
            "doc/page/login.asp", "login.asp", "login.html",
        ]
        return sum(1 for i in indicators if i in cl) >= 2

    # ------------------------------------------------------------------------
    #  Credential testing
    # ------------------------------------------------------------------------
    def _verify_creds(self, ip, port, user, pw, session=None):
        if not self.verify_creds:
            return True
        for ep in VERIFY_ENDPOINTS:
            url = self._build_url(ip, port, ep)
            code, content, hdrs = self._fetch(
                url, username=user, password=pw, session=session
            )
            if hdrs.get("_error"):
                continue
            if code == 200 and content and len(content) > 50:
                cl = content.lower()
                if "login" not in cl[:300]:
                    return True
        return False

    def _test_single_cred(self, ip, port, user, pw, session=None):
        url = self._build_url(ip, port)
        if not _HAS_REQUESTS:
            code, content, hdrs = self._fetch(url, username=user, password=pw, session=session)
            if hdrs.get("_error"):
                return False, False, "none"
            if code in (401, 403):
                return False, False, "none"
            if code == 200:
                verified = self._verify_creds(ip, port, user, pw, session=session)
                return True, verified, "basic_or_digest"
            return False, False, "none"

        from requests.auth import HTTPBasicAuth, HTTPDigestAuth
        sess = session or self._get_session()
        
        try:
            resp = sess.request(
                "GET", url, auth=HTTPBasicAuth(user, pw or ""),
                timeout=self.timeout, allow_redirects=True, verify=False,
            )
        except (requests.RequestException, OSError, ssl.SSLError):
            return False, False, "none"
        except Exception:
            return False, False, "none"
            
        if resp.status_code == 200:
            verified = self._verify_creds(ip, port, user, pw, session=sess)
            return True, verified, "basic"
            
        if resp.status_code == 401:
            www_auth = resp.headers.get("WWW-Authenticate", "").lower()
            if "digest" in www_auth:
                try:
                    resp_d = sess.request(
                        "GET", url, auth=HTTPDigestAuth(user, pw or ""),
                        timeout=self.timeout, allow_redirects=True, verify=False,
                    )
                except (requests.RequestException, OSError, ssl.SSLError):
                    return False, False, "none"
                except Exception:
                    return False, False, "none"
                    
                if resp_d.status_code == 200:
                    verified = self._verify_creds(ip, port, user, pw, session=sess)
                    return True, verified, "digest"
        return False, False, "none"

    def _test_creds_parallel(self, ip, port, creds_list, session=None):
        result = [None, None, "none", False]
        result_lock = threading.Lock()
        stop_event = threading.Event()

        def _worker(user, pw):
            if stop_event.is_set():
                return
            found, verified, auth_type = self._test_single_cred(
                ip, port, user, pw, session=session
            )
            if found:
                with result_lock:
                    if result[0] is None:
                        if verified:
                            result[0], result[1], result[2], result[3] = user, pw, auth_type, True
                            stop_event.set()
                        else:
                            result[0], result[1], result[2], result[3] = user, pw, auth_type, False

        with ThreadPoolExecutor(max_workers=self.cred_threads) as ex:
            futs = {ex.submit(_worker, u, p): (u, p) for u, p in creds_list}
            for f in as_completed(futs):
                if stop_event.is_set():
                    break
        if result[0]:
            return tuple(result)
        return (None, None, "none", False)

    # ------------------------------------------------------------------------
    #  RTSP verification
    # ------------------------------------------------------------------------
    def _test_rtsp(self, ip, brand, user, pw):
        if not _HAS_CV2:
            return self._test_rtsp_socket(ip, brand, user, pw)
        pattern = RTSP_PATTERNS.get(brand, RTSP_PATTERNS["generic"])
        url = pattern.format(user=user, pw=pw, ip=ip)
        try:
            cap = cv2.VideoCapture(url)
            if cap is None or not cap.isOpened():
                return False
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout * 1000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout * 1000)
            except AttributeError:
                pass
            ret = cap.isOpened()
            if ret:
                ret, _ = cap.read()
            cap.release()
            return ret
        except (cv2.error, Exception):
            return False

    def _test_rtsp_socket(self, ip, brand, user, pw):
        """Raw-socket fallback with OPTIONS probe."""
        pattern = RTSP_PATTERNS.get(brand, RTSP_PATTERNS["generic"])
        url = pattern.format(user=user, pw=pw, ip=ip)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((str(ip), 554))
            
            auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
            
            # 1. Send OPTIONS probe
            req_options = (
                f"OPTIONS {url} RTSP/1.0\r\n"
                f"CSeq: 1\r\n"
                f"Authorization: Basic {auth}\r\n"
                f"User-Agent: CCTVScanner\r\n"
                f"\r\n"
            )
            s.sendall(req_options.encode())
            resp_options = s.recv(4096).decode("utf-8", errors="ignore")
            
            if "401 Unauthorized" in resp_options:
                s.close()
                return False
                
            # 2. Send DESCRIBE
            req_describe = (
                f"DESCRIBE {url} RTSP/1.0\r\n"
                f"CSeq: 2\r\n"
                f"Accept: application/sdp\r\n"
                f"Authorization: Basic {auth}\r\n"
                f"User-Agent: CCTVScanner\r\n"
                f"\r\n"
            )
            s.sendall(req_describe.encode())
            resp_describe = s.recv(4096).decode("utf-8", errors="ignore")
            s.close()
            
            return "200 OK" in resp_describe
        except (socket.timeout, socket.error, ConnectionRefusedError, OSError):
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------------
    #  Config endpoint check
    # ------------------------------------------------------------------------
    def _check_config(self, ip, port, brand, user=None, pw=None, session=None):
        details = {}
        eps = BRAND_ENDPOINTS.get(brand, BRAND_ENDPOINTS["generic"])
        for ep in eps:
            url = self._build_url(ip, port, ep)
            code, content, hdrs = self._fetch(
                url, username=user, password=pw, session=session
            )
            details[ep] = {
                "status": code,
                "content_type": hdrs.get("Content-Type", ""),
                "length": len(content) if content else 0,
                "error": hdrs.get("_error"),
            }
        return details

    # ------------------------------------------------------------------------
    #  Single-host scan
    # ------------------------------------------------------------------------
    def _scan_host(self, ip):
        start_time = time.time()
        r = ScanResult(ip=str(ip))
        ip_str = str(ip)
        ports_to_scan = [80, 443, 554] if self.quick else list(CAMERA_PORTS.keys())
        prog = self._progress()

        if self.rate_limiter:
            self.rate_limiter.acquire()

        safe_print(f"  [{prog}/{len(self.targets)}] {ip_str} - Scanning ports...", end="", flush=True)
        port_results = self._scan_ports_parallel(ip_str, ports_to_scan)
        
        for pr in port_results:
            if pr.is_open:
                r.open_ports.append(pr.port)
            r.port_scan_details.append({
                "port": pr.port,
                "open": pr.is_open,
                "error": pr.error,
            })

        if not r.open_ports:
            safe_print(f" No open ports")
            r.scan_duration = time.time() - start_time
            return r

        safe_print(f" Found {len(r.open_ports)} open port(s): {r.open_ports}")
        http_port = 80 if 80 in r.open_ports else r.open_ports[0]

        r.detected_brand, r.brand_source = self._detect_brand(ip_str, http_port)
        safe_print(f"  Brand: {r.detected_brand} ({r.brand_source})")

        url = self._build_url(ip_str, http_port)
        session = self._get_session()
        code, content, hdrs = self._fetch(url, session=session)
        if not hdrs.get("_error"):
            r.http_banner = hdrs.get("Server", hdrs.get("server", "Unknown"))
            r.login_page_detected = self._is_login_page(code, content)

        creds_list = self._get_credentials_for_brand(r.detected_brand)
        # Fix: Do not pass session to worker threads - requests.Session is not thread-safe
        # Each worker thread will create its own thread-local session
        user, pw, auth_type, is_verified = self._test_creds_parallel(
            ip_str, http_port, creds_list, session=None
        )
        
        if user is not None:
            r.default_login_found = (user, pw)
            r.auth_type_used = auth_type
            r.credentials_verified = is_verified
            if r.credentials_verified:
                safe_print(f"  LOGIN VERIFIED ({auth_type}): {user}:{pw}")
            else:
                safe_print(f"  Auth accepted for {user}:{pw} ({auth_type}) but NOT verified")

        if r.credentials_verified and self.check_rtsp and 554 in r.open_ports:
            if self._test_rtsp(ip_str, r.detected_brand, user, pw):
                r.rtsp_accessible = True
                safe_print(f"  RTSP stream accessible")

        if r.credentials_verified:
            r.config_details = self._check_config(
                ip_str, http_port, r.detected_brand, user, pw, session=None
            )
            r.default_config_accessible = any(
                d.get("status") == 200 for d in r.config_details.values()
                if isinstance(d, dict) and not d.get("error")
            )
            if r.default_config_accessible:
                safe_print(f"  Config endpoints accessible")

        r.scan_duration = time.time() - start_time
        return r

    # ------------------------------------------------------------------------
    #  Main scan loop
    # ------------------------------------------------------------------------
    def scan(self):
        safe_print("")
        safe_print("=" * 70)
        safe_print("  CCTV / IP Camera Scanner v10")
        safe_print("=" * 70)
        safe_print(f"  Targets      : {len(self.targets)} hosts")
        safe_print(f"  Timeout      : {self.timeout}s")
        safe_print(f"  Threads      : {self.threads}")
        safe_print(f"  Cred Threads : {self.cred_threads}")
        safe_print(f"  Quick mode   : {'Yes' if self.quick else 'No'}")
        safe_print(f"  Verify creds : {'Yes' if self.verify_creds else 'No'}")
        safe_print(f"  Check RTSP   : {'Yes' if self.check_rtsp else 'No'}")
        safe_print(f"  Use ONVIF    : {'Yes' if self.use_onvif else 'No'}")
        safe_print(f"  Rate limit   : {self.rate_limiter.rate}/sec" if self.rate_limiter else "  Rate limit   : Off")
        safe_print(f"  Custom creds : {'Yes' if self.creds_file else 'No'}")
        safe_print(f"  requests     : {'Yes' if _HAS_REQUESTS else 'No (urllib+Digest fallback)'}")
        safe_print(f"  OpenCV       : {'Yes' if _HAS_CV2 else 'No (socket fallback)'}")
        safe_print("=" * 70)
        safe_print("")

        start = time.time()
        self._executor = ThreadPoolExecutor(max_workers=self.threads)
        try:
            futs = {self._executor.submit(self._scan_host, ip): ip for ip in self.targets}
            for f in as_completed(futs):
                try:
                    self.results.append(f.result())
                except Exception as e:
                    safe_print(f"  Error scanning {futs[f]}: {e}")
        finally:
            if self._executor:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # Python < 3.9 does not support cancel_futures parameter
                    self._executor.shutdown(wait=False)
                self._executor = None

        elapsed = time.time() - start
        safe_print("")
        safe_print("=" * 70)
        safe_print(f"  Done in {elapsed:.1f}s")
        with_ports = [r for r in self.results if r.open_ports]
        verified = [r for r in self.results if r.credentials_verified]
        safe_print(f"  Hosts with open ports   : {len(with_ports)}")
        safe_print(f"  Verified default logins : {len(verified)}")
        safe_print("=" * 70)
        safe_print("")
        return self.results

    # ------------------------------------------------------------------------
    #  Reporting
    # ------------------------------------------------------------------------
    def print_report(self):
        safe_print("")
        safe_print("=" * 70)
        safe_print("  DETAILED REPORT")
        safe_print("=" * 70)
        safe_print("")

        cams = [r for r in self.results if r.open_ports]
        if not cams:
            safe_print("  No cameras/devices found.")
            safe_print("")
            return

        verified = [r for r in cams if r.credentials_verified]
        unverified = [r for r in cams if r.default_login_found and not r.credentials_verified]
        clean = [r for r in cams if not r.default_login_found]

        def _ports_str(ports):
            parts = []
            for p in ports:
                label = CAMERA_PORTS.get(p, "Unknown")
                parts.append(f"{p} ({label})")
            return ", ".join(parts)

        if verified:
            safe_print("  [VERIFIED DEFAULT CREDENTIALS]")
            safe_print("  " + "-" * 60)
            for r in verified:
                safe_print(f"  IP        : {r.ip}")
                safe_print(f"  Ports     : {_ports_str(r.open_ports)}")
                safe_print(f"  Brand     : {r.detected_brand} ({r.brand_source})")
                safe_print(f"  Banner    : {r.http_banner or 'N/A'}")
                safe_print(f"  Login Pg  : {'Yes' if r.login_page_detected else 'No'}")
                safe_print(f"  Credentials (VERIFIED, {r.auth_type_used}): {r.default_login_found[0]}:{r.default_login_found[1]}")
                if r.rtsp_accessible:
                    safe_print(f"  RTSP      : Accessible")
                if r.default_config_accessible:
                    safe_print(f"  Config    : Accessible")
                    for ep, d in r.config_details.items():
                        if isinstance(d, dict) and d.get("status") == 200 and not d.get("error"):
                            safe_print(f"            - {ep}")
                safe_print(f"  Scan time : {r.scan_duration:.2f}s")
                safe_print("")

        if unverified:
            safe_print("  [UNVERIFIED - Likely False Positives]")
            safe_print("  " + "-" * 60)
            for r in unverified:
                safe_print(f"  IP        : {r.ip}")
                safe_print(f"  Ports     : {_ports_str(r.open_ports)}")
                safe_print(f"  Brand     : {r.detected_brand} ({r.brand_source})")
                safe_print(f"  Attempted : {r.default_login_found[0]}:{r.default_login_found[1]} ({r.auth_type_used})")
                safe_print(f"  Status    : NOT VERIFIED")
                safe_print(f"  Scan time : {r.scan_duration:.2f}s")
                safe_print("")

        if clean:
            safe_print("  [CAMERAS FOUND - No Default Login]")
            safe_print("  " + "-" * 60)
            for r in clean:
                safe_print(f"  IP        : {r.ip}")
                safe_print(f"  Ports     : {_ports_str(r.open_ports)}")
                safe_print(f"  Brand     : {r.detected_brand} ({r.brand_source})")
                safe_print(f"  Login Pg  : {'Yes' if r.login_page_detected else 'No'}")
                safe_print(f"  Scan time : {r.scan_duration:.2f}s")
                safe_print("")

        safe_print("=" * 70)
        safe_print("  SUMMARY")
        safe_print("=" * 70)
        safe_print(f"  Cameras found             : {len(cams)}")
        safe_print(f"  Verified default creds    : {len(verified)}")
        safe_print(f"  Unverified login attempts : {len(unverified)}")
        safe_print(f"  No default login          : {len(clean)}")
        safe_print("")

        if verified:
            safe_print("  [!] VERIFIED VULNERABILITIES")
            safe_print("  1. Change default passwords IMMEDIATELY")
            safe_print("  2. Update firmware")
            safe_print("  3. Disable unused services (RTSP, ONVIF)")
            safe_print("  4. Separate camera VLAN")
            safe_print("  5. Disable exposed config endpoints")
            safe_print("  6. Enable HTTPS, disable HTTP")
        elif unverified:
            safe_print("  [?] Unverified attempts - may be false positives")
            safe_print("  1. Manually test the credentials in a browser")
            safe_print("  2. Camera may use Digest auth (not Basic)")
            safe_print("  3. Use --no-verify to see all attempts")
        else:
            safe_print("  [OK] No default credential vulnerabilities found.")
        safe_print("")
        safe_print("=" * 70)
        safe_print("")

    def save_report_json(self, filepath: str):
        report = {
            "scan_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "targets": len(self.targets),
                "timeout": self.timeout,
                "threads": self.threads,
                "cred_threads": self.cred_threads,
                "quick_mode": self.quick,
                "verify_creds": self.verify_creds,
                "check_rtsp": self.check_rtsp,
                "use_onvif": self.use_onvif,
                "rate_limit": self.rate_limiter.rate if self.rate_limiter else 0,
                "custom_creds_file": self.creds_file,
                "requests_available": _HAS_REQUESTS,
                "opencv_available": _HAS_CV2,
            },
            "results": [],
        }
        for r in self.results:
            if r.open_ports:
                report["results"].append({
                    "ip": r.ip,
                    "open_ports": r.open_ports,
                    "port_details": r.port_scan_details,
                    "brand": r.detected_brand,
                    "brand_source": r.brand_source,
                    "login_page": r.login_page_detected,
                    "default_login": list(r.default_login_found) if r.default_login_found else None,
                    "auth_type": r.auth_type_used,
                    "verified": r.credentials_verified,
                    "rtsp": r.rtsp_accessible,
                    "config_accessible": r.default_config_accessible,
                    "config_details": r.config_details,
                    "http_banner": r.http_banner,
                    "scan_duration": r.scan_duration,
                    "errors": r.errors,
                })
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        safe_print(f"[+] Detailed report saved to {filepath}")

# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="CCTV/IP Camera Scanner v10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s -n 192.168.1.0/24\n"
            "  %(prog)s -n 192.168.1.0/24 -q\n"
            "  %(prog)s -t 192.168.1.100\n"
            "  %(prog)s -n 192.168.1.0/24 --rate-limit 2\n"
            "  %(prog)s -n 192.168.1.0/24 --creds-file my_creds.json\n"
            "  %(prog)s -n 192.168.1.0/24 --no-verify\n"
            "\n"
            "Custom credentials file format (JSON):\n"
            '{\n'
            '  "hikvision": [["admin", "mypass"]],\n'
            '  "generic": [["user", "pass"]]\n'
            '}\n'
            "\n"
            "Optional dependencies (recommended):\n"
            "  pip install requests opencv-python\n"
        ),
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-n", "--network", help="Network CIDR, e.g. 192.168.1.0/24")
    g.add_argument("-t", "--target", help="Single IP, e.g. 192.168.1.100")
    ap.add_argument("-q", "--quick", action="store_true", help="Only ports 80,443,554")
    ap.add_argument("--timeout", type=int, default=3, help="Socket timeout (default 3)")
    ap.add_argument("--threads", type=int, default=30, help="Host scan threads (default 30)")
    ap.add_argument("--cred-threads", type=int, default=6, help="Credential threads per host (default 6)")
    ap.add_argument("-o", "--output", help="Save report to JSON file")
    ap.add_argument("--no-verify", action="store_true", help="Skip credential verification")
    ap.add_argument("--no-rtsp", action="store_true", help="Skip RTSP verification")
    ap.add_argument("--no-onvif", action="store_true", help="Skip ONVIF brand detection")
    ap.add_argument("--rate-limit", type=float, default=0,
                    help="Max requests per second (0 = unlimited)")
    ap.add_argument("--creds-file", help="JSON file with custom credentials")

    args = ap.parse_args()

    scanner = CCTVScanner(
        network=args.network,
        target=args.target,
        timeout=args.timeout,
        threads=args.threads,
        cred_threads=args.cred_threads,
        quick=args.quick,
        verify_creds=not args.no_verify,
        check_rtsp=not args.no_rtsp,
        use_onvif=not args.no_onvif,
        rate_limit=args.rate_limit,
        creds_file=args.creds_file,
    )

    def signal_handler(sig, frame):
        safe_print("\n[!] Interrupt received. Saving partial report and shutting down...")
        if scanner._executor:
            try:
                scanner._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 does not support cancel_futures parameter
                scanner._executor.shutdown(wait=False)
        if args.output:
            scanner.save_report_json(args.output)
        else:
            scanner.save_report_json("scan_report_partial.json")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    scanner.scan()
    scanner.print_report()

    if args.output:
        scanner.save_report_json(args.output)

if __name__ == "__main__":
    main()