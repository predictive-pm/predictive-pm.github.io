"""Convert the PM·TWIN Design-canvas artboard (.dc.html) into a standalone
static web app (Vue 3, no build step) for GitHub Pages.

usage: python build_webapp.py <Main.dc.html> <ui_data.json> <out_dir>
"""
import re, sys, os, shutil, json

src, data_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
xdc = re.search(r"<x-dc>(.*)</x-dc>", s, re.S).group(1)
helmet = re.search(r"<helmet>(.*?)</helmet>", xdc, re.S).group(1)
markup = xdc.replace(re.search(r"<helmet>.*?</helmet>", xdc, re.S).group(0), "")
script = re.search(r'<script type="text/x-dc" data-dc-script[^>]*>(.*?)</script>', s, re.S).group(1)

LIT = re.compile(r"^(true|false|null|-?\d+(\.\d+)?)$")
scope: list[str] = []


def expr(e: str) -> str:
    e = e.strip()
    if LIT.match(e):
        return e
    head = e.split(".")[0]
    if head in scope or head == "$index":
        return e
    return "v." + e


def conv_attr_value(val: str) -> tuple[bool, str]:
    """returns (is_bound, js_expression_or_literal)"""
    holes = list(re.finditer(r"\{\{(.*?)\}\}", val))
    if not holes:
        return False, val
    if len(holes) == 1 and holes[0].group(0) == val.strip():
        return True, expr(holes[0].group(1))
    parts, last = [], 0
    for h in holes:
        lit = val[last:h.start()].replace("`", "\\`").replace("${", "\\${")
        parts.append(lit + "${" + expr(h.group(1)) + "}")
        last = h.end()
    parts.append(val[last:].replace("`", "\\`"))
    return True, "`" + "".join(parts) + "`"


ATTR = re.compile(r'([:@\w-]+)(?:="([^"]*)"|=\'([^\']*)\')?')
TAG = re.compile(r"<(/?)([a-zA-Z][\w-]*)((?:[^>\"']|\"[^\"]*\"|'[^']*')*?)(/?)>")


def conv_tag(m):
    close, name, attrs, selfc = m.group(1), m.group(2), m.group(3), m.group(4)
    if name == "sc-for":
        if close:
            scope.pop(); return "</template>"
        a = dict((k, v1 or v2) for k, v1, v2 in ATTR.findall(attrs))
        lst = conv_attr_value(a["list"])[1]
        var = a["as"]
        out_ = f'<template v-for="({var}, $index) in ({lst} || [])">'
        scope.append(var)
        return out_
    if name == "sc-if":
        if close:
            return "</template>"
        a = dict((k, v1 or v2) for k, v1, v2 in ATTR.findall(attrs))
        return f'<template v-if="{conv_attr_value(a["value"])[1]}">'
    if close:
        return m.group(0)
    new = []
    for am in ATTR.finditer(attrs):
        k, v1, v2 = am.group(1), am.group(2), am.group(3)
        val = v1 if v1 is not None else v2
        if k.startswith("hint-"):
            continue
        if val is None:
            new.append(k); continue
        bound, js = conv_attr_value(val)
        if k.startswith("on") and len(k) > 2 and k[2].isupper():
            ev = k[2:].lower()
            if ev == "change" and name == "input":
                ev = "input"
            new.append(f'@{ev}="{js}"'); continue
        if bound:
            new.append(f':{k}="{js}"')
        else:
            new.append(f'{k}="{val}"')
    return "<" + name + ("" if not new else " " + " ".join(new)) + (" /" if selfc else "") + ">"


def conv_text(t: str) -> str:
    return re.sub(r"\{\{(.*?)\}\}", lambda h: "{{ " + expr(h.group(1)) + " }}", t)


out_parts, last = [], 0
for m in TAG.finditer(markup):
    out_parts.append(conv_text(markup[last:m.start()]))
    out_parts.append(conv_tag(m))
    last = m.end()
out_parts.append(conv_text(markup[last:]))
template = "".join(out_parts)
template = re.sub(r"<!--.*?-->", "", template, flags=re.S)
assert not scope, scope

# data URL -> local file
script = re.sub(r"const DATA_URL = '[^']+';", "const DATA_URL = './data/pm_twin_data.json';", script)

html = f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PM·TWIN — Predictive Maintenance · เครื่องล้างหม้อหุงและฝา</title>
<meta name="description" content="Digital Twin + AI (LSTM-Autoencoder) Predictive Maintenance dashboard for the pot & lid washing machine.">
<meta property="og:title" content="PM·TWIN — Predictive Maintenance Digital Twin">
<meta property="og:description" content="ทำนายความล้มเหลวของอุปกรณ์รายส่วน · LSTM-Autoencoder · Digital Twin">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Cpolygon points='20,2 36,11 36,29 20,38 4,29 4,11' fill='%230A0E13' stroke='%2322D3EE' stroke-width='3'/%3E%3Ccircle cx='20' cy='20' r='5' fill='%2322D3EE'/%3E%3C/svg%3E">
{helmet}
<style>[v-cloak]{{display:none}}</style>
</head>
<body>
<div id="app" v-cloak></div>
<script type="text/x-template" id="tpl">
{template}
</script>
<script src="./vendor/vue.global.prod.js"></script>
<script>
class DCLogic {{
  constructor(props) {{ this.props = props || {{}}; this.state = {{}}; }}
  setState(o) {{
    Object.assign(this.state, typeof o === 'function' ? o(this.state) : o);
    if (this.__onUpdate) this.__onUpdate();
  }}
  forceUpdate() {{ if (this.__onUpdate) this.__onUpdate(); }}
}}
{script}
(function () {{
  const comp = new Component({{}});
  const v = Vue.shallowRef(comp.renderVals());
  let queued = false;
  comp.__onUpdate = function () {{
    if (queued) return; queued = true;
    Promise.resolve().then(function () {{ queued = false; v.value = comp.renderVals(); }});
  }};
  Vue.createApp({{ setup: function () {{ return {{ v: v }}; }}, template: '#tpl' }}).mount('#app');
  comp.componentDidMount();
}})();
</script>
</body>
</html>
"""
os.makedirs(f"{out}/data", exist_ok=True); os.makedirs(f"{out}/vendor", exist_ok=True)
open(f"{out}/index.html", "w", encoding="utf-8").write(html)
shutil.copy(data_json, f"{out}/data/pm_twin_data.json")
shutil.copy("node_modules/vue/dist/vue.global.prod.js", f"{out}/vendor/vue.global.prod.js")
open(f"{out}/.nojekyll", "w").write("")
print("ok", len(html))
