if (window.__REPORTS_JS_LOADED__) {
  console.warn("reports.js loaded twice; ignoring second load");
} else {
  window.__REPORTS_JS_LOADED__ = true;

  let CHARTS = {};
  let LAST_REPORT = null;

  let RDA = {
    calories: 0,
    protein: 0,
    carbs: 0,
    fat: 0,
    fiber: 0,
    sodium: 0
  };

  async function loadRDA() {
    const urls = ["/api/metrics", "/data/metrics.json"];

    for (const url of urls) {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) continue;
        const data = await res.json();

        const t = data?.targets ?? data ?? {};
        RDA = {
          calories: Number(t.calories ?? 0),
          protein: Number(t.protein_g ?? t.protein ?? 0),
          carbs: Number(t.carbs_g ?? t.carbs ?? 0),
          fat: Number(t.fat_g ?? t.fat ?? 0),
          fiber: Number(t.fiber_g ?? t.fiber ?? 0),
          sodium: Number(t.sodium_mg ?? t.sodium ?? 0),
        };

        return RDA;
      } catch (e) {
      }
    }

    return RDA;
  }

 
  function rdaForMetric(metricKey){
    if (!metricKey) return null;
    const map = {
      calories_kcal: "calories",
      protein_g: "protein",
      carbs_g: "carbs",
      fat_g: "fat",
      fiber_g: "fiber",
      sodium_mg: "sodium",
    };
    const k = map[metricKey];
    if (!k) return null;
    const v = RDA[k];
    return (v === null || v === undefined) ? null : v;
  }


  const MACRO_CAL = { carbs_g: 4, protein_g: 4, fat_g: 9 };

  function $(id){ return document.getElementById(id); }

  function isoDate(d){
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth()+1).padStart(2,"0");
    const dd = String(d.getDate()).padStart(2,"0");
    return `${yyyy}-${mm}-${dd}`;
  }

  function parseISODate(s){
    const [y,m,d] = s.split("-").map(Number);
    return new Date(y, (m||1)-1, d||1, 0,0,0,0);
  }

  function setStatus(msg){
    const el = $("status");
    if (el) el.textContent = msg;
  }

  function destroyChart(key){
    if (CHARTS[key]) { CHARTS[key].destroy(); delete CHARTS[key]; }
  }

  function fmt(n){
    if (n === null || n === undefined || Number.isNaN(n)) return "";
    const abs = Math.abs(n);
    if (abs >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (abs >= 10) return n.toLocaleString(undefined, { maximumFractionDigits: 1 });
    return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  function daysBetweenInclusive(start, end){
    const ms = 24*60*60*1000;
    const a = new Date(start.getFullYear(), start.getMonth(), start.getDate()).getTime();
    const b = new Date(end.getFullYear(), end.getMonth(), end.getDate()).getTime();
    return Math.floor((b - a)/ms) + 1;
  }

  function rollingAverage(series, window){
    const out = [];
    const w = Math.max(1, window|0);
    let sum = 0;
    const q = [];
    for (let i=0;i<series.length;i++){
      const y = series[i].y ?? 0;
      q.push(y);
      sum += y;
      if (q.length > w) sum -= q.shift();
      out.push({ x: series[i].x, y: sum / q.length });
    }
    return out;
  }

  function bucketKey(dateObj, mode){
    const y = dateObj.getFullYear();
    const m = dateObj.getMonth()+1;
    if (mode === "month") return `${y}-${String(m).padStart(2,"0")}`;
    if (mode === "year") return `${y}`;
    const q = Math.floor((m-1)/3)+1;
    return `${y}-Q${q}`;
  }

  function groupSum(points, mode){
    const map = new Map();
    for (const p of points){
      const k = bucketKey(p.date, mode);
      map.set(k, (map.get(k) || 0) + (p.y || 0));
    }
    const keys = [...map.keys()].sort((a,b)=> a.localeCompare(b));
    return keys.map(k => ({ x:k, y: map.get(k) }));
  }

  async function fetchReport(startISO, endISO){
    const url = `/api/report?start=${encodeURIComponent(startISO)}&end=${encodeURIComponent(endISO)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Report fetch failed: ${res.status}`);
    return await res.json();
  }

  function normalizeMetrics(report){
    const totals = report.totals || {};
    const daily = report.daily || [];

    const set = new Set(Object.keys(totals));
    for (const d of daily){
      const m = d.metrics || {};
      Object.keys(m).forEach(k=>set.add(k));
    }
    const preferred = ["calories_kcal","protein_g","carbs_g","fat_g","fiber_g","sugar_g","sodium_mg"];
    const rest = [...set].filter(k=>!preferred.includes(k)).sort();
    const metrics = [...preferred.filter(k=>set.has(k)), ...rest];

    return { metrics, totals, daily };
  }

  function renderTotalsTable(metrics, totals, startISO, endISO){
    const start = parseISODate(startISO);
    const end = parseISODate(endISO);
    const ndays = daysBetweenInclusive(start, end);
    $("rangeLabel").textContent = `${startISO} → ${endISO} (${ndays} day${ndays===1?"":"s"})`;

    const tbody = $("totalsTable").querySelector("tbody");
    tbody.innerHTML = "";

    for (const k of metrics){
      const total = totals[k] ?? 0;
      const avg = total / ndays;
      const rda = rdaForMetric(k);
      const pct = (rda && rda !== 0) ? (total / (rda * ndays)) * 100 : null;

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${k}</td>
        <td>${fmt(total)}</td>
        <td>${fmt(avg)}</td>
        <td>${rda==null ? "" : fmt(rda)}</td>
        <td>${pct==null ? "" : fmt(pct)}${pct==null ? "" : "%"}</td>
      `;
      tbody.appendChild(tr);
    }
  }

  function renderMacroPie(totals){
    const carbs = totals.carbs_g ?? 0;
    const protein = totals.protein_g ?? 0;
    const fat = totals.fat_g ?? 0;

    const carbCal = carbs * MACRO_CAL.carbs_g;
    const proteinCal = protein * MACRO_CAL.protein_g;
    const fatCal = fat * MACRO_CAL.fat_g;

    const data = [carbCal, proteinCal, fatCal];
    const labels = ["Carbs (4)","Protein (4)","Fat (9)"];

    destroyChart("macroPie");
    const ctx = $("macroPie").getContext("2d");
    CHARTS.macroPie = new Chart(ctx, {
      type: "pie",
      data: { labels, datasets: [{ data }] },
      options: { responsive: true, plugins: { legend: { display: true } } }
    });

    const totalCal = data.reduce((a,b)=>a+b,0) || 1;
    $("macroLegend").innerHTML = `
      <div><b>Macro calories</b></div>
      <div>Carbs: ${fmt(carbCal)} kcal (${fmt(100*carbCal/totalCal)}%)</div>
      <div>Protein: ${fmt(proteinCal)} kcal (${fmt(100*proteinCal/totalCal)}%)</div>
      <div>Fat: ${fmt(fatCal)} kcal (${fmt(100*fatCal/totalCal)}%)</div>
      <div class="muted" style="margin-top:6px">Note: macro-derived calories may not exactly match labeled calories.</div>
    `;
  }

  function renderHourlyLine(report, metric){
    const arr = getHourlySeriesFromReport(report, metric);
    const unit = metricUnit(metric);
    const yTitle = unit ? `${metric} (${unit})` : metric;

    destroyChart("hourlyLine");
    const ctx = $("hourlyLine").getContext("2d");
    CHARTS.hourlyLine = new Chart(ctx, {
      type: "line",
      data: {
        labels: Array.from({length:24}, (_,h)=> String(h).padStart(2,"0")),
        datasets: [{ label: `Avg ${yTitle}`, data: arr }]
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { title: { display:true, text:"Hour" } },
          y: { title: { display:true, text: yTitle } }
        }
      }
    });
  }

  function buildDailySeries(daily, metric){
    const rows = daily.slice().sort((a,b)=> a.date.localeCompare(b.date));
    return rows.map(r => ({
      date: parseISODate(r.date),
      x: r.date,
      y: (r.metrics && (r.metrics[metric] ?? 0)) ?? 0
    }));
  }

  function renderTrendCharts(daily, metric, rollDays){
    const series = buildDailySeries(daily, metric);
    const roll = rollingAverage(series, rollDays);

    destroyChart("dailyTrend");
    CHARTS.dailyTrend = new Chart($("dailyTrend").getContext("2d"), {
      type: "line",
      data: {
        labels: series.map(p=>p.x),
        datasets: [
          { label: `${metric} (daily)`, data: series.map(p=>p.y) },
          { label: `${metric} (${rollDays}d rolling avg)`, data: roll.map(p=>p.y) },
        ]
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { title: { display:true, text:"Date" } },
          y: { title: { display:true, text: metric } }
        }
      }
    });

    const monthAgg = groupSum(series, "month");
    const qAgg = groupSum(series, "quarter");
    const yAgg = groupSum(series, "year");

    destroyChart("monthlyBar");
    CHARTS.monthlyBar = new Chart($("monthlyBar").getContext("2d"), {
      type:"bar",
      data:{ labels: monthAgg.map(p=>p.x), datasets:[{ label: metric, data: monthAgg.map(p=>p.y) }] },
      options:{ responsive:true }
    });

    destroyChart("quarterlyBar");
    CHARTS.quarterlyBar = new Chart($("quarterlyBar").getContext("2d"), {
      type:"bar",
      data:{ labels: qAgg.map(p=>p.x), datasets:[{ label: metric, data: qAgg.map(p=>p.y) }] },
      options:{ responsive:true }
    });

    destroyChart("annualBar");
    CHARTS.annualBar = new Chart($("annualBar").getContext("2d"), {
      type:"bar",
      data:{ labels: yAgg.map(p=>p.x), datasets:[{ label: metric, data: yAgg.map(p=>p.y) }] },
      options:{ responsive:true }
    });
  }

  function renderStatsDump(report, metrics, totals, daily){
    const ndays = daily.length;
    const metricCoverage = {};
    for (const k of metrics){
      let present = 0;
      for (const d of daily){
        const v = d.metrics ? d.metrics[k] : undefined;
        if (v !== null && v !== undefined) present++;
      }
      metricCoverage[k] = { present_days: present, missing_days: Math.max(0, ndays - present) };
    }

    const dump = {
      range: { start: report.start, end: report.end },
      days_returned: ndays,
      entries_count: report.entries_count ?? null,
      metrics_count: metrics.length,
      has_hourly_avg: !!report.hourly_avg,
      totals_keys: Object.keys(totals).length,
      metric_coverage: metricCoverage,
      meta: report.meta ?? null
    };

    $("statsDump").textContent = JSON.stringify(dump, null, 2);
  }

  function setPresetDates(preset){
    const today = new Date();
    const end = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    let start = null;

    if (preset === "custom") return;
    if (preset === "all") {
      $("startDate").value = "";
      $("endDate").value = isoDate(end);
      return;
    }
    if (preset === "ytd") {
      start = new Date(end.getFullYear(), 0, 1);
    } else {
      const map = { "7d":7, "30d":30, "90d":90, "180d":180, "365d":365 };
      const n = map[preset] || 30;
      start = new Date(end.getTime() - (n-1)*24*60*60*1000);
    }

    $("startDate").value = isoDate(start);
    $("endDate").value = isoDate(end);
  }

  function populateTrendMetric(metrics){
    const sel = $("trendMetric");
    const cur = sel.value;
    sel.innerHTML = "";
    for (const m of metrics){
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    }
    if (metrics.includes(cur)) sel.value = cur;
    else if (metrics.includes("calories_kcal")) sel.value = "calories_kcal";
    else sel.value = metrics[0] || "";
  }


  function metricUnit(metric){
    if (!metric) return "";
    if (metric.endsWith("_kcal") || metric === "calories_kcal") return "kcal";
    if (metric.endsWith("_mg") || metric === "sodium_mg") return "mg";
    if (metric.endsWith("_g")) return "g";
    return "";
  }

  function populateHourlyMetric(metrics){
    const sel = $("hourlyMetric");
    if (!sel) return;
    const cur = sel.value;

    const preferred = ["calories_kcal","protein_g","carbs_g","fat_g","fiber_g","sugar_g","sodium_mg"];
    const list = [
      ...preferred.filter(k=>metrics.includes(k)),
      ...metrics.filter(k=>!preferred.includes(k))
    ];

    sel.innerHTML = "";
    for (const m of list){
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    }

    if (list.includes(cur)) sel.value = cur;
    else if (list.includes("calories_kcal")) sel.value = "calories_kcal";
    else sel.value = list[0] || "";
  }


  const HOURLY_GUIDANCE = {
    calories_kcal: {
      title: "Calories (calories_kcal)",
      best: [
        "Gentle rise in the morning, steadier mid-day, taper in the evening",
        "Smaller variance day-to-day (fewer huge spikes)"
      ],
      flags: [
        "Big spike after ~7–8pm (often stress/reward eating)",
        "Very low until late afternoon, then cramming calories late"
      ],
      note: "Calories are a summary metric. If the shape looks off, it’s usually meal timing or food choices underneath."
    },
    protein_g: {
      title: "Protein (protein_g)",
      best: [
        "Spread fairly evenly across meals/snacks",
        "Similar-sized protein “hits” rather than one giant dinner mountain"
      ],
      flags: [
        "Most protein at dinner only",
        "Protein flatlined until noon or later"
      ],
      note: "Even distribution improves satiety and supports muscle retention during weight loss."
    },
    carbs_g: {
      title: "Carbs (carbs_g)",
      best: [
        "More earlier and mid-day, smaller in the evening",
        "Carbs paired with protein/fiber to avoid sharp swings"
      ],
      flags: [
        "Largest carb spike late evening (dessert/snack pattern)",
        "Carbs arriving mostly as liquid sugar or low-fiber foods"
      ],
      note: "This is not “no carbs at night.” It’s “don’t make 9pm your biggest carb event.”"
    },
    fat_g: {
      title: "Fat (fat_g)",
      best: [
        "Moderate and spread out through the day",
        "Often paired with protein for steadier appetite"
      ],
      flags: [
        "Huge fat spike late evening (cheese, oils, nuts, “healthy snacks” that aren’t)",
        "Fat spikes that quietly blow up total calories"
      ],
      note: "Big late fat loads can push calories up fast, even if the food feels small."
    },
    fiber_g: {
      title: "Fiber (fiber_g)",
      best: [
        "Distributed across meals, not saved for one sitting",
        "Often slightly higher earlier in the day"
      ],
      flags: [
        "All fiber at dinner (can worsen sleep/GI comfort)",
        "Fiber nearly absent most of the day"
      ],
      note: "Consistent fiber improves appetite control and digestion regularity."
    },
    sugar_g: {
      title: "Sugar (sugar_g)",
      best: [
        "Low overall, and if present it’s earlier in the day",
        "Sugar mostly comes with fiber/protein (fruit, dairy) rather than isolated"
      ],
      flags: [
        "Evening sugar spikes (dessert, stress, liquid calories)",
        "Sugar bumps without fiber/protein alongside"
      ],
      note: "Late sugar is rarely planned. It’s usually convenience or coping."
    },
    sodium_mg: {
      title: "Sodium (sodium_mg)",
      best: [
        "Fairly even distribution",
        "Slightly higher earlier if you train or sweat"
      ],
      flags: [
        "Massive sodium at dinner (processed food territory)",
        "Sodium spikes without corresponding calories often point to jerky, sauces, or packaged snacks"
      ],
      note: "If you see sodium peaks without protein, that’s snack engineering, not nutrition."
    }
  };

  function renderHourlyGuidance(metric){
    const el = $("hourlyGuidance");
    if (!el) return;

    const g = HOURLY_GUIDANCE[metric] || { title: metric, best: [], flags: [], note: "" };

    const bestLis = (g.best || []).map(t=> `<li>${escapeHtml(t)}</li>`).join("");
    const flagLis = (g.flags || []).map(t=> `<li>${escapeHtml(t)}</li>`).join("");
    const note = g.note ? `<div class="note">${escapeHtml(g.note)}</div>` : "";

    el.innerHTML = `
      <h3>${escapeHtml(g.title || metric)}</h3>
      <div class="subhead">Best case</div>
      <ul>${bestLis || "<li>(no guidance yet)</li>"}</ul>
      <div class="subhead">Red flags</div>
      <ul>${flagLis || "<li>(none listed)</li>"}</ul>
      ${note}
    `;
  }

  function getHourlySeriesFromReport(report, metric){
    const pick = (obj)=>{
      if (!obj) return null;
      if (Array.isArray(obj)) return obj.slice(0,24);
      if (typeof obj === "object"){
        return Array.from({length:24}, (_,h)=> obj[String(h)] ?? obj[h] ?? 0);
      }
      return null;
    };

    const ha = report && (report.hourly_avg ?? report.hourlyAvg ?? report.hourly);
    if (ha && typeof ha === "object" && !Array.isArray(ha)) {
      const candidates = [
        ha[metric],
        (report.hourly_avg_metrics && report.hourly_avg_metrics[metric]),
        (report.hourly_by_metric && report.hourly_by_metric[metric]),
        (report.hourly && report.hourly[metric]),
      ];
      for (const c of candidates){
        const p = pick(c);
        if (p) return p;
      }
    }

    const legacy = pick(ha);
    if (legacy && metric === "calories_kcal") return legacy;

    return Array.from({length:24}, ()=>0);
  }

  async function run(){
    try{
      setStatus("Loading…");

      const preset = $("preset").value;
      if (preset !== "custom") setPresetDates(preset);

      const startISO = $("startDate").value;
      const endISO = $("endDate").value || isoDate(new Date());

      if (!endISO) throw new Error("End date required.");
      if (preset !== "all" && !startISO) throw new Error("Start date required (or use All time).");

      const report = await fetchReport(startISO || "", endISO);
      LAST_REPORT = report;

      const { metrics, totals, daily } = normalizeMetrics(report);

      populateTrendMetric(metrics);
      renderTotalsTable(metrics, totals, report.start, report.end);
      renderMacroPie(totals);
      populateHourlyMetric(metrics);
      const hMetric = ($("hourlyMetric") && $("hourlyMetric").value) ? $("hourlyMetric").value : "calories_kcal";
      renderHourlyLine(report, hMetric);
      renderHourlyGuidance(hMetric);
      renderStatsDump(report, metrics, totals, daily);

      const metric = $("trendMetric").value;
      const rollDays = parseInt($("rollDays").value || "7", 10);
      renderTrendCharts(daily, metric, rollDays);

      setStatus(`Done. Metrics: ${metrics.length}, days: ${daily.length}`);
    } catch (e){
      console.error(e);
      setStatus(`Error: ${e.message || e}`);
    }
  }

  function exportJSON(){
    if (!LAST_REPORT) { setStatus("Nothing to export."); return; }
    const blob = new Blob([JSON.stringify(LAST_REPORT, null, 2)], { type:"application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `report_${LAST_REPORT.start || "start"}_${LAST_REPORT.end || "end"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function wire(){
    const btnRun = $("btnRun");
    const btnExport = $("btnExport");
    if (!btnRun || !btnExport) {
      setStatus("Missing controls in DOM (btnRun/btnExport).");
      return;
    }

    btnRun.onclick = run;
    btnExport.onclick = exportJSON;

    $("preset").onchange = ()=> setPresetDates($("preset").value);

    $("trendMetric").onchange = ()=> {
      if (!LAST_REPORT) return;
      const { daily } = normalizeMetrics(LAST_REPORT);
      const metric = $("trendMetric").value;
      const rollDays = parseInt($("rollDays").value || "7", 10);
      renderTrendCharts(daily, metric, rollDays);
    };


    $("hourlyMetric").onchange = ()=> {
      if (!LAST_REPORT) return;
      const metric = $("hourlyMetric").value || "calories_kcal";
      renderHourlyLine(LAST_REPORT, metric);
      renderHourlyGuidance(metric);
    };

    $("rollDays").onchange = ()=> {
      if (!LAST_REPORT) return;
      const { daily } = normalizeMetrics(LAST_REPORT);
      const metric = $("trendMetric").value;
      const rollDays = parseInt($("rollDays").value || "7", 10);
      renderTrendCharts(daily, metric, rollDays);
    };

    $("preset").value = "30d";
    setPresetDates("30d");
    setStatus("Ready.");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", async () => {
      await loadRDA();
      wire();
    });
  } else {
    loadRDA().finally(wire);
  }
}

function escapeHtml(s){
  return String(s ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

