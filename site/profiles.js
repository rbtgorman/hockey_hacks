/* Shared by profiles.html, team.html and player.html.
   Everything drawn here comes from site/data/profiles_index.json (the metric
   registry, league figures, team and player lists) and one profile file.
   Metrics are drawn in registry order, so a new metric shows up with no page change. */

const $ = (id) => document.getElementById(id);

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (v != null) e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return r.json();
}

const loadIndex = () => getJSON("data/profiles_index.json");

function fmt(v, kind) {
  if (v == null || Number.isNaN(v)) return "–";
  switch (kind) {
    case "int": return Math.round(v).toLocaleString("en-US");
    case "dec1": return v.toFixed(1);
    case "dec2": return v.toFixed(2);
    case "pct": return (v * 100).toFixed(1) + "%";
    case "signed": {
      // Round first, so a tiny negative shows as 0.0 rather than −0.0.
      const r = Math.round(v * 10) / 10;
      return (r > 0 ? "+" : r < 0 ? "−" : "") + Math.abs(r).toFixed(1);
    }
    case "min": return Math.round(v).toLocaleString("en-US") + " min";
    default: return String(v);
  }
}

const SAMPLE_WORDS = {
  gp: "games", toi: "minutes", toi_5v5: "5-on-5 minutes", toi_pp: "power-play minutes",
  toi_pk: "penalty-kill minutes", att: "unblocked shots", sa: "unblocked shots faced",
};
const sampleText = (key, n) => `${Math.round(n).toLocaleString("en-US")} ${SAMPLE_WORDS[key] || key}`;

function seasonInProgress(index, season) {
  return !!index.seasons.find((s) => s.season === season)?.in_progress;
}

/* One tile per metric, grouped under the registry's headings. */
function renderTiles(root, metrics, row, league) {
  root.replaceChildren();
  const groups = [];
  for (const m of metrics) {
    let g = groups.find((x) => x.name === m.group);
    if (!g) groups.push((g = { name: m.group, items: [] }));
    g.items.push(m);
  }
  for (const g of groups) {
    const grid = h("div", { class: "stats" });
    for (const m of g.items) {
      const v = row.values[m.key];
      const n = row.counts[m.sample] ?? 0;
      const thin = m.min > 0 && n < m.min;
      const lgv = league ? league[m.key] : null;
      let lg = null, vs = null;
      if (lgv != null && v != null) {
        lg = h("span", { class: "lg" }, "League average: " + fmt(lgv, m.fmt));
        // Better or worse than the league average, by any amount (not a rank).
        // Only when the sample is big enough and the two differ at the
        // precision shown.
        if (m.better && !thin) {
          if (fmt(v, m.fmt) === fmt(lgv, m.fmt)) vs = h("span", { class: "vs eq" }, "Same as league average");
          else {
            const good = m.better === "high" ? v > lgv : v < lgv;
            vs = h("span", { class: "vs " + (good ? "up" : "dn") },
              good ? "Better than league average" : "Worse than league average");
          }
        }
      }
      const smp = thin
        ? `Only ${sampleText(m.sample, n)}. Needs ${m.min.toLocaleString("en-US")} before this means much.`
        : m.min > 0 ? `Based on ${sampleText(m.sample, n)}` : null;
      grid.append(h("div", { class: "stat" + (thin ? " thin" : "") },
        h("span", { class: "k" }, m.label),
        h("span", { class: "v" }, fmt(v, m.fmt)),
        lg,
        vs,
        h("span", { class: "d" }, m.define),
        smp ? h("span", { class: "smp" }, smp) : null));
    }
    root.append(h("section", {}, h("div", { class: "section-head" }, h("h2", {}, g.name)), grid));
  }
}

/* Shot zones: attempts, share of all attempts, goals, xG, xG per shot, with a
   league share to compare against when one is given. */
function renderZones(root, zoneDefs, zones, leagueZones, caption) {
  const total = Object.values(zones).reduce((s, z) => s + z[0], 0);
  const ltotal = leagueZones ? Object.values(leagueZones).reduce((s, z) => s + z[0], 0) : 0;
  const max = Math.max(0.01, ...zoneDefs.map((d) => (zones[d.key]?.[0] || 0) / (total || 1)),
    ...zoneDefs.map((d) => leagueZones ? (leagueZones[d.key]?.[0] || 0) / (ltotal || 1) : 0));
  const head = ["Where from", "Shots", "Share", leagueZones ? "League share" : null, "Goals", "xG", "xG per shot"]
    .filter(Boolean).map((t) => h("th", {}, t));
  const body = zoneDefs.map((d) => {
    const [a, g, x] = zones[d.key] || [0, 0, 0];
    const sh = total ? a / total : 0;
    const cells = [
      h("td", { title: d.define }, d.label),
      h("td", {}, a.toLocaleString("en-US")),
      h("td", {}, h("span", { class: "bar", style: `width:${Math.round(60 * sh / max)}px` }), " ", (sh * 100).toFixed(1) + "%"),
    ];
    if (leagueZones) {
      const ls = ltotal ? (leagueZones[d.key]?.[0] || 0) / ltotal : 0;
      cells.push(h("td", {}, h("span", { class: "bar lg", style: `width:${Math.round(60 * ls / max)}px` }), " ", (ls * 100).toFixed(1) + "%"));
    }
    cells.push(h("td", {}, String(g)), h("td", {}, x.toFixed(1)), h("td", {}, a ? (x / a).toFixed(3) : "–"));
    return h("tr", {}, ...cells);
  });
  root.replaceChildren(
    h("p", { class: "hint" }, caption),
    h("div", { class: "tablewrap" }, h("table", {}, h("thead", {}, h("tr", {}, ...head)), h("tbody", {}, ...body))),
    h("details", {}, h("summary", {}, "How the zones are drawn"),
      h("ul", {}, ...zoneDefs.map((d) => h("li", {}, h("b", {}, d.label + ": "), d.define)))));
}

/* Every season side by side for a few headline metrics. */
function renderSeasons(root, metrics, keys, seasons) {
  const ms = keys.map((k) => metrics.find((m) => m.key === k)).filter(Boolean);
  const head = h("tr", {}, h("th", {}, "Season"), ...ms.map((m) => h("th", { title: m.define }, m.label)));
  const rows = seasons.map((s) => h("tr", {}, h("td", {}, s.label),
    ...ms.map((m) => {
      const thin = m.min > 0 && (s.counts[m.sample] ?? 0) < m.min;
      return h("td", { class: thin ? "thin" : null }, fmt(s.values[m.key], m.fmt));
    })));
  root.replaceChildren(h("div", { class: "tablewrap" },
    h("table", {}, h("thead", {}, head), h("tbody", {}, ...rows))));
}

function seasonSelect(sel, seasons, current, onChange) {
  sel.replaceChildren(...seasons.map((s) => h("option", { value: s.season }, s.label)));
  sel.value = String(current);
  sel.addEventListener("change", () => onChange(Number(sel.value)));
}

function setParam(name, value) {
  const u = new URL(location.href);
  u.searchParams.set(name, value);
  history.replaceState(null, "", u);
}

function showError(root, msg) {
  root.replaceChildren(h("p", { class: "note warn" }, msg));
}

const POS_NAMES = { C: "Center", L: "Left wing", R: "Right wing", D: "Defense", G: "Goalie" };

function updatedLine(index) {
  const d = new Date(index.generated_at);
  return "Updated " + d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
}

/* The note every profile shows about the model behind the xG. */
const XG_NOTE_HTML = "<p class=\"note\"><strong>About the xG here.</strong> It comes from the shot model (<span class=\"mv\"></span>), which doesn't know who's shooting or who's in net yet, and it's due to be retrained. Treat it as a first pass. Every number says what it's based on, and the ones with too little behind them are greyed out.</p>";
