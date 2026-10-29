"""Sinh các test plan JMeter (.jmx) cho hệ thống HAUI Fitness.

Tham số runtime truyền bằng -J (vd: -Jthreads=100 -Jramp=20 -Jduration=60 -Jhost=127.0.0.1).
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "jmeter"


def P(name, default):
    return f"${{__P({name},{default})}}"


def sp(name, value):
    return f'<stringProp name="{name}">{escape(str(value))}</stringProp>'


def bp(name, value):
    return f'<boolProp name="{name}">{"true" if value else "false"}</boolProp>'


def http(name, method, path, port_prop="port", port_default="8000", body=None, embedded=False,
         timeout_resp="60000", extra=""):
    body_xml = ""
    if body is not None:
        body_xml = (
            '<boolProp name="HTTPSampler.postBodyRaw">true</boolProp>'
            '<elementProp name="HTTPsampler.Arguments" elementType="Arguments"><collectionProp name="Arguments.arguments">'
            '<elementProp name="" elementType="HTTPArgument">'
            + bp("HTTPArgument.always_encode", False) + sp("Argument.value", body) + sp("Argument.metadata", "=")
            + "</elementProp></collectionProp></elementProp>"
        )
    else:
        body_xml = ('<elementProp name="HTTPsampler.Arguments" elementType="Arguments">'
                    '<collectionProp name="Arguments.arguments"/></elementProp>')
    emb = ""
    if embedded:
        emb = (bp("HTTPSampler.image_parser", True) + bp("HTTPSampler.concurrentDwn", True)
               + sp("HTTPSampler.concurrentPool", "6")
               + sp("HTTPSampler.embedded_url_re", r"http://[^/]*100\.95\.228\.82.*|http://127\.0\.0\.1.*"))
    return (
        f'<HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="{escape(name)}" enabled="true">'
        + body_xml
        + sp("HTTPSampler.domain", P("host", "100.95.228.82"))
        + sp("HTTPSampler.port", P(port_prop, port_default))
        + sp("HTTPSampler.protocol", "http")
        + sp("HTTPSampler.path", path)
        + sp("HTTPSampler.method", method)
        + bp("HTTPSampler.follow_redirects", True)
        + bp("HTTPSampler.use_keepalive", True)
        + sp("HTTPSampler.connect_timeout", "10000")
        + sp("HTTPSampler.response_timeout", timeout_resp)
        + emb + extra
        + "</HTTPSamplerProxy>"
    )


def headers(pairs):
    items = "".join(
        f'<elementProp name="" elementType="Header">{sp("Header.name", k)}{sp("Header.value", v)}</elementProp>'
        for k, v in pairs
    )
    return (f'<HeaderManager guiclass="HeaderPanel" testclass="HeaderManager" testname="Headers" enabled="true">'
            f'<collectionProp name="HeaderManager.headers">{items}</collectionProp></HeaderManager>')


def json_extract(var, path, default="NOTFOUND"):
    return (f'<JSONPostProcessor guiclass="JSONPostProcessorGui" testclass="JSONPostProcessor" testname="Extract {var}" enabled="true">'
            + sp("JSONPostProcessor.referenceNames", var) + sp("JSONPostProcessor.jsonPathExprs", path)
            + sp("JSONPostProcessor.match_numbers", "1") + sp("JSONPostProcessor.defaultValues", default)
            + "</JSONPostProcessor>")


def csv_ds(file, vars_):
    return ('<CSVDataSet guiclass="TestBeanGUI" testclass="CSVDataSet" testname="Accounts" enabled="true">'
            + sp("filename", file) + sp("fileEncoding", "UTF-8") + sp("variableNames", vars_)
            + bp("ignoreFirstLine", False) + sp("delimiter", ",") + bp("quotedData", False)
            + bp("recycle", True) + bp("stopThread", False) + sp("shareMode", "shareMode.all")
            + "</CSVDataSet>")


def think(ms_prop="think", default="0"):
    return ('<ConstantTimer guiclass="ConstantTimerGui" testclass="ConstantTimer" testname="Think time" enabled="true">'
            + sp("ConstantTimer.delay", P(ms_prop, default)) + "</ConstantTimer>")


def wrap(children):
    """children: list of (element_xml, [children])"""
    out = ""
    for el, kids in children:
        out += el + "<hashTree>" + wrap(kids) + "</hashTree>"
    return out


def plan(name, tg_children, plan_children=()):
    tg = (
        f'<ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="{escape(name)} users" enabled="true">'
        + sp("ThreadGroup.on_sample_error", "continue")
        + '<elementProp name="ThreadGroup.main_controller" elementType="LoopController" guiclass="LoopControlPanel" testclass="LoopController">'
        + bp("LoopController.continue_forever", False) + sp("LoopController.loops", P("loops", "-1")) + "</elementProp>"
        + sp("ThreadGroup.num_threads", P("threads", "10"))
        + sp("ThreadGroup.ramp_time", P("ramp", "10"))
        + bp("ThreadGroup.scheduler", True)
        + sp("ThreadGroup.duration", P("duration", "60"))
        + sp("ThreadGroup.delay", "0")
        + bp("ThreadGroup.same_user_on_next_iteration", True)
        + "</ThreadGroup>"
    )
    tp = (f'<TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="{escape(name)}" enabled="true">'
          + bp("TestPlan.functional_mode", False) + bp("TestPlan.serialize_threadgroups", False)
          + '<elementProp name="TestPlan.user_defined_variables" elementType="Arguments"><collectionProp name="Arguments.arguments"/></elementProp>'
          + "</TestPlan>")
    tree = [(tp, list(plan_children) + [(tg, tg_children)])]
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<jmeterTestPlan version="1.2" properties="5.0" jmeter="5.6.3">'
            "<hashTree>" + wrap(tree) + "</hashTree></jmeterTestPlan>")


JSON_H = headers([("Content-Type", "application/json")])
AUTH_H = headers([("Content-Type", "application/json"), ("Authorization", "Bearer ${token}")])
LOGIN_BODY = '{"email":"${email}","password":"${__P(password)}"}'
ACCOUNTS = csv_ds(P("accounts", "accounts.csv"), "email")


def once_only(kids):
    el = ('<OnceOnlyController guiclass="OnceOnlyControllerGui" testclass="OnceOnlyController" '
          'testname="Login once per user" enabled="true"/>')
    return (el, kids)


def build():
    OUT.mkdir(exist_ok=True)
    # 1) FE: tải trang chính kèm toàn bộ CSS/JS/ảnh (như trình duyệt lần đầu, không cache)
    regex = ('<RegexExtractor guiclass="RegexExtractorGui" testclass="RegexExtractor" testname="Extract assets" enabled="true">'
             + sp("RegexExtractor.useHeaders", "false") + sp("RegexExtractor.refname", "asset")
             + sp("RegexExtractor.regex", r'(?:src|href)="((?:chunk|main|styles|polyfills)-[A-Za-z0-9]+\.(?:js|css)|assets/[^"]+\.(?:png|svg|jpg|ico))"')
             + sp("RegexExtractor.template", "$1$") + sp("RegexExtractor.default", "")
             + sp("RegexExtractor.match_number", "-1") + "</RegexExtractor>")
    foreach = ('<ForeachController guiclass="ForeachControlPanel" testclass="ForeachController" testname="Each asset" enabled="true">'
               + sp("ForeachController.inputVal", "asset") + sp("ForeachController.returnVal", "a")
               + bp("ForeachController.useSeparator", True) + "</ForeachController>")
    trans = ('<TransactionController guiclass="TransactionControllerGui" testclass="TransactionController" testname="FE PAGE LOAD (html+css+js)" enabled="true">'
             + bp("TransactionController.includeTimers", False) + bp("TransactionController.parent", False)
             + "</TransactionController>")
    (OUT / "fe_page.jmx").write_text(plan("FE page load", [
        (trans, [
            (http("FE GET / (index.html)", "GET", "/", "port", "4200"), [(regex, [])]),
            (foreach, [(http("FE GET asset", "GET", "/${a}", "port", "4200"), [])]),
        ]),
        (think(), []),
    ]))
    # 2) BE: 1 endpoint đơn (baseline), path truyền bằng -Jpath
    (OUT / "be_simple.jmx").write_text(plan("BE simple", [
        (http("GET ${__P(path,/)}", "GET", P("path", "/"), "port", "8000"), []),
        (think(), []),
    ]))
    # 3) BE login (bcrypt, tốn CPU)
    (OUT / "be_login.jmx").write_text(plan("BE login", [
        (ACCOUNTS, []), (JSON_H, []),
        (http("POST /api/auth/login", "POST", "/api/auth/login", body=LOGIN_BODY), []),
        (think(), []),
    ]))
    # 4) BE user flow: login 1 lần rồi lặp các API sinh viên hay gọi
    (OUT / "be_user_flow.jmx").write_text(plan("BE student flow", [
        (ACCOUNTS, []),
        once_only([
            (http("POST /api/auth/login", "POST", "/api/auth/login", body=LOGIN_BODY), [
                (JSON_H, []), (json_extract("token", "$.data.accessToken"), [])]),
        ]),
        (AUTH_H, []),
        (http("GET /api/auth/me", "GET", "/api/auth/me"), []),
        (http("GET /api/enrollments/my-courses", "GET", "/api/enrollments/my-courses"), []),
        (http("GET /api/tasks", "GET", "/api/tasks?page_size=20"), []),
        (http("GET /api/video-results", "GET", "/api/video-results?page_size=10"), [
            (json_extract("rid", "$.data.items[0].id"), [])]),
        (http("GET /api/video-results/{id}/status", "GET", "/api/video-results/${rid}/status"), []),
        (http("GET /api/sports", "GET", "/api/sports"), []),
        (http("GET /api/notifications", "GET", "/api/notifications"), []),
        (think(), []),
    ]))
    # 4b) Giống 4 nhưng dùng token đăng nhập sẵn (tokens.csv: email,token) để không tạo bão login
    (OUT / "be_user_flow_tok.jmx").write_text(plan("BE student flow (pre-auth)", [
        (csv_ds(P("tokens", "tokens.csv"), "email,token"), []),
        (AUTH_H, []),
        (http("GET /api/auth/me", "GET", "/api/auth/me"), []),
        (http("GET /api/enrollments/my-courses", "GET", "/api/enrollments/my-courses"), []),
        (http("GET /api/tasks", "GET", "/api/tasks?page_size=20"), []),
        (http("GET /api/video-results", "GET", "/api/video-results?page_size=10"), [
            (json_extract("rid", "$.data.items[0].id"), [])]),
        (http("GET /api/video-results/{id}/status", "GET", "/api/video-results/${rid}/status"), []),
        (http("GET /api/sports", "GET", "/api/sports"), []),
        (http("GET /api/notifications", "GET", "/api/notifications"), []),
        (think(), []),
    ]))
    # 4c) Giáo viên: đăng nhập 1 lần rồi xem sổ điểm / thống kê / bài nộp (truy vấn nặng)
    course = P("course", "6ae093a7bf50c057267d76e7")
    task = P("task", "6ae093a7bf50c057267d76e9")
    teacher_body = '{"email":"${__P(teacher,longdv@haui.edu.vn)}","password":"${__P(teacher_password)}"}'
    (OUT / "teacher_flow.jmx").write_text(plan("Teacher flow", [
        once_only([
            (http("POST /api/auth/login (teacher)", "POST", "/api/auth/login", body=teacher_body), [
                (JSON_H, []), (json_extract("token", "$.data.accessToken"), [])]),
        ]),
        (AUTH_H, []),
        (http("GET /teacher/gradebook", "GET", f"/api/teacher/courses/{course}/gradebook"), []),
        (http("GET /teacher/stats", "GET", f"/api/teacher/courses/{course}/stats"), []),
        (http("GET /teacher/course tasks", "GET", f"/api/teacher/courses/{course}/tasks"), []),
        (http("GET /tasks/{id}/submissions", "GET", f"/api/tasks/{task}/submissions"), []),
        (think(), []),
    ]))
    # 5) AI trực tiếp: mỗi request 1 job_id mới (tránh cache kết quả của AI)
    ai_body = (
        '{"job_id":"jm-${__UUID()}","result_id":"loadtest","task_id":"t","course_id":"c","sport_id":"s",'
        '"sport_code":"PICKLEBALL","mode":"VIDEO","bucket":"videos","object_key":"${__P(object_key,x)}",'
        '"video_url":"http://unused","minio_endpoint":"${__P(minio,100.95.228.82:9000)}","user_id":"loadtest",'
        '"attempt":1,"schema_version":1,"detail":{"title":"loadtest","category":"practice",'
        '"gradingMethod":"ai","timeLimit":"60"}}'
    )
    (OUT / "ai_direct.jmx").write_text(plan("AI direct", [
        (JSON_H, []),
        (http("POST /api/v1/process", "POST", "/api/v1/process", "aiport", "8888", body=ai_body,
              timeout_resp="1800000"), []),
        (think(), []),
    ]))
    (OUT / "accounts.csv").write_text("\n".join([
        "annv@sv.haui.edu.vn", "binhtt@sv.haui.edu.vn", "cuonglh@sv.haui.edu.vn", "ducpm@sv.haui.edu.vn",
        "hoaht@sv.haui.edu.vn", "huynq@sv.haui.edu.vn", "lanvt@sv.haui.edu.vn", "minhdt@sv.haui.edu.vn",
        "ngocbt@sv.haui.edu.vn", "phongdq@sv.haui.edu.vn"]) + "\n")
    print("written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    build()
