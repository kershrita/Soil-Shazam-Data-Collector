
/* ─── Shared utilities ──────────────────────────────────────────────────── */
function escapeHtml(s) {
  if (s === undefined || s === null) return "";
  var div = document.createElement("div");
  div.textContent = String(s);
  return div.innerHTML;
}

function formatCat(cat) {
  return String(cat || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, function (c) { return c.toUpperCase(); });
}

function handleBrokenImage(img) {
  img.onerror = null;
  img.src = "";
  img.alt = "Image not found";
  img.classList.add("img-broken");
}

/* Image browser logic for the step detail page. */

/* Distribution chart */
(function () {
  var COLOR_PALETTE = [
    "#357560", "#384727", "#F5D54A", "#CABDAF",
    "#527462", "#3A887C", "#C0C49B", "#D5DCDC",
    "#B87F45", "#9EB45B", "#7D8D87", "#CBC497",
    "#D3B73E", "#8B341F", "#89A68F", "#A3A47A",
  ];

  function valueColor(val) {
    var h = 0;
    for (var i = 0; i < val.length; i++) h = (h * 31 + val.charCodeAt(i)) & 0x7fffffff;
    return COLOR_PALETTE[h % COLOR_PALETTE.length];
  }

  window.renderDistributionChart = function (canvas, distributions, removedDistributions) {
    if (!canvas || !distributions) return;
    var categories = Object.keys(distributions);
    if (!categories.length) return;

    var seen = {};
    categories.forEach(function (cat) {
      Object.keys(distributions[cat]).forEach(function (val) { seen[val] = true; });
      if (removedDistributions && removedDistributions[cat]) {
        Object.keys(removedDistributions[cat]).forEach(function (val) { seen[val] = true; });
      }
    });
    var allValues = Object.keys(seen);

    var datasets = [];
    allValues.forEach(function (val) {
      datasets.push({
        label: val,
        data: categories.map(function (cat) { return distributions[cat][val] || 0; }),
        backgroundColor: valueColor(val),
        borderWidth: 0,
        stack: "kept",
      });
    });

    if (removedDistributions) {
      allValues.forEach(function (val) {
        datasets.push({
          label: val + " (removed)",
          data: categories.map(function (cat) {
            return (removedDistributions[cat] && removedDistributions[cat][val]) || 0;
          }),
          backgroundColor: valueColor(val) + "55",
          borderWidth: 1,
          borderColor: valueColor(val),
          borderDash: [3, 3],
          stack: "removed",
        });
      });
    }

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: categories.map(function (c) {
          var s = c.replace(/_/g, " ");
          return s.charAt(0).toUpperCase() + s.slice(1);
        }),
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { stacked: true, ticks: { font: { size: 11 }, color: "#6b7280" }, grid: { display: false } },
          y: { stacked: true, ticks: { font: { size: 11 }, color: "#6b7280" }, grid: { color: "#e0e3e8" } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: "index",
            filter: function (item) { return item.parsed.y > 0; },
            callbacks: {
              label: function (ctx) { return ctx.dataset.label + ": " + ctx.parsed.y; },
            },
          },
        },
      },
    });
  };
}());


(function () {
  "use strict";

  var state = {
    currentPage: 1,
    perPage: 60,
    density: "comfy",
    totalPages: 1,
    currentImages: [],
    currentIndex: -1,
    debounceTimer: null,
    detailCache: new Map(),
    activeDetailToken: null,
    handlingPopstate: false,
    lastFocused: null,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function parsePositiveInt(value, fallback) {
    var parsed = parseInt(value, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  }

  function getSearchInput() {
    return byId("search-input");
  }

  function getPerPageSelect() {
    return byId("per-page-select");
  }

  function getDensitySelect() {
    return byId("density-select");
  }

  function getGrid() {
    return byId("image-grid");
  }

  function getOverlay() {
    return byId("detail-overlay");
  }

  function isOverlayOpen() {
    var overlay = getOverlay();
    return !!overlay && !overlay.classList.contains("hidden");
  }

  function applyDensityClass() {
    var grid = getGrid();
    if (!grid) return;
    grid.classList.remove("density-comfy", "density-compact");
    grid.classList.add(state.density === "compact" ? "density-compact" : "density-comfy");
  }

  function setDetailLoading(isLoading) {
    var loading = byId("detail-loading");
    var panel = document.querySelector(".detail-panel");
    if (loading) loading.classList.toggle("hidden", !isLoading);
    if (panel) panel.classList.toggle("is-loading", !!isLoading);
  }

  function buildUiParams(includeImage) {
    var params = new URLSearchParams();
    params.set("page", String(state.currentPage));
    params.set("per_page", String(state.perPage));
    params.set("density", state.density);

    if (window.VIEW_MODE) params.set("view", window.VIEW_MODE);

    var searchInput = getSearchInput();
    if (searchInput && searchInput.value.trim()) {
      params.set("search", searchInput.value.trim());
    }

    document.querySelectorAll(".filter-select[data-category]").forEach(function (sel) {
      if (sel.value) params.set(sel.dataset.category, sel.value);
    });

    if (includeImage && state.currentIndex >= 0 && state.currentImages[state.currentIndex]) {
      params.set("image", state.currentImages[state.currentIndex].filename);
    }

    return params;
  }

  function syncUrl(mode, includeImage) {
    if (state.handlingPopstate) return;
    var params = buildUiParams(includeImage !== false);
    var query = params.toString();
    var nextUrl = window.location.pathname + (query ? ("?" + query) : "");
    var currentUrl = window.location.pathname + window.location.search;
    if (nextUrl === currentUrl) return;

    if (mode === "push") {
      history.pushState({}, "", nextUrl);
    } else {
      history.replaceState({}, "", nextUrl);
    }
  }

  function applyUrlState() {
    var params = new URLSearchParams(window.location.search);
    state.currentPage = parsePositiveInt(params.get("page"), 1);

    var perPage = parsePositiveInt(params.get("per_page"), 60);
    if ([30, 60, 120].indexOf(perPage) === -1) perPage = 60;
    state.perPage = perPage;

    var density = params.get("density");
    state.density = density === "compact" ? "compact" : "comfy";

    var searchInput = getSearchInput();
    if (searchInput) searchInput.value = params.get("search") || "";

    var perPageSelect = getPerPageSelect();
    if (perPageSelect) perPageSelect.value = String(state.perPage);

    var densitySelect = getDensitySelect();
    if (densitySelect) densitySelect.value = state.density;

    document.querySelectorAll(".filter-select[data-category]").forEach(function (sel) {
      var value = params.get(sel.dataset.category) || "";
      var hasOption = Array.from(sel.options).some(function (opt) { return opt.value === value; });
      sel.value = hasOption ? value : "";
    });
  }

  function buildApiParams() {
    var params = new URLSearchParams();
    params.set("page", String(state.currentPage));
    params.set("per_page", String(state.perPage));
    if (window.VIEW_MODE) params.set("view", window.VIEW_MODE);

    var searchInput = getSearchInput();
    if (searchInput && searchInput.value.trim()) {
      params.set("search", searchInput.value.trim());
    }

    document.querySelectorAll(".filter-select[data-category]").forEach(function (sel) {
      if (sel.value) params.set(sel.dataset.category, sel.value);
    });

    return params.toString();
  }

  function showGridSkeleton() {
    var grid = getGrid();
    if (!grid) return;
    grid.classList.add("is-loading");
    applyDensityClass();
    var count = Math.min(state.perPage, 24);
    var html = "";
    for (var i = 0; i < count; i++) {
      html +=
        '<div class="image-skeleton">' +
        '<div class="image-skeleton-thumb"></div>' +
        '<div class="image-skeleton-meta">' +
        '<div class="image-skeleton-line"></div>' +
        '<div class="image-skeleton-line short"></div>' +
        "</div></div>";
    }
    grid.innerHTML = html;
    grid.setAttribute("aria-busy", "true");
  }

  function addTag(wrap, tagTpl, text, extraClass) {
    if (tagTpl && tagTpl.content) {
      var t = tagTpl.content.cloneNode(true);
      var span = t.querySelector(".label-tag");
      span.textContent = text;
      if (extraClass) span.classList.add(extraClass);
      wrap.appendChild(t);
    } else {
      var span2 = document.createElement("span");
      span2.className = "label-tag" + (extraClass ? (" " + extraClass) : "");
      span2.textContent = text;
      wrap.appendChild(span2);
    }
  }

  function buildLabelTags(wrap, img, tagTpl) {
    if (img.rejection) {
      addTag(wrap, tagTpl, img.rejection.reason === "overlay" ? "Overlay" : "Low score", "rejection-tag");
      var scoreText = img.rejection.reason === "overlay"
        ? "ov=" + (img.rejection.overlay_score || "")
        : "pos=" + (img.rejection.positive_score || "");
      addTag(wrap, tagTpl, scoreText, "score-tag");
      if (img.source) addTag(wrap, tagTpl, img.source, "score-tag");
      return;
    }
    if (img.removed) {
      addTag(wrap, tagTpl, "Removed", "rejection-tag");
      if (img.source) addTag(wrap, tagTpl, img.source, "score-tag");
      return;
    }
    if (img.labels) {
      Object.keys(img.labels).forEach(function (cat) {
        if (img.labels[cat]) addTag(wrap, tagTpl, img.labels[cat], "");
      });
    }
    if (img.source) addTag(wrap, tagTpl, img.source, "score-tag");
  }

  function renderGrid(data) {
    var grid = getGrid();
    if (!grid) return;
    grid.classList.remove("is-loading");
    grid.removeAttribute("aria-busy");
    applyDensityClass();

    if (!data.images.length) {
      grid.innerHTML = '<div class="empty-state">No images found.</div>';
      return;
    }

    var tpl = byId("tpl-image-card");
    var tagTpl = byId("tpl-label-tag");
    var frag = document.createDocumentFragment();

    data.images.forEach(function (img, idx) {
      var clone = tpl && tpl.content ? tpl.content.cloneNode(true) : null;
      var card = clone ? clone.querySelector(".image-card") : document.createElement("div");

      if (!clone) {
        card.className = "image-card";
        card.innerHTML =
          '<img loading="lazy" alt="">' +
          '<div class="image-card-info">' +
          '<div class="image-card-name"></div>' +
          '<div class="image-card-labels"></div>' +
          "</div>";
      }

      card.dataset.idx = idx;
      card.dataset.filename = img.filename;
      card.setAttribute("role", "button");
      card.setAttribute("tabindex", "0");

      if (img.rejection || img.removed) card.classList.add("rejected");

      var displayName = img.filename.split("/").pop();
      var imgEl = card.querySelector("img");
      imgEl.src = img.thumb_url || img.url;
      imgEl.alt = displayName;
      imgEl.onload = function () { this.classList.add("loaded"); };
      imgEl.onerror = function () { handleBrokenImage(this); };

      var nameEl = card.querySelector(".image-card-name");
      nameEl.textContent = displayName;
      nameEl.title = img.filename;

      var labelsWrap = card.querySelector(".image-card-labels");
      labelsWrap.innerHTML = "";
      buildLabelTags(labelsWrap, img, tagTpl);

      card.setAttribute("aria-label", "Open details for " + displayName);
      card.addEventListener("click", function () {
        openDetail(idx, { historyMode: "push" });
      });
      card.addEventListener("keydown", function (evt) {
        if (evt.key === "Enter" || evt.key === " ") {
          evt.preventDefault();
          openDetail(idx, { historyMode: "push" });
        }
      });

      if (clone) frag.appendChild(clone);
      else frag.appendChild(card);
    });

    grid.innerHTML = "";
    grid.appendChild(frag);
  }

  function renderPagination(data) {
    var el = byId("pagination");
    if (!el) return;
    if (data.pages <= 1) {
      el.innerHTML = "";
      return;
    }

    var html = "";
    if (state.currentPage > 1) {
      html += '<button class="btn btn-small" data-page="' + (state.currentPage - 1) + '">&larr;</button>';
    }

    var startPage = Math.max(1, state.currentPage - 4);
    var endPage = Math.min(data.pages, startPage + 8);
    startPage = Math.max(1, endPage - 8);

    for (var p = startPage; p <= endPage; p++) {
      var active = p === state.currentPage ? " active" : "";
      html += '<button class="btn btn-small' + active + '" data-page="' + p + '">' + p + "</button>";
    }

    if (state.currentPage < data.pages) {
      html += '<button class="btn btn-small" data-page="' + (state.currentPage + 1) + '">&rarr;</button>';
    }

    html += '<span class="page-info">' + data.total + " images</span>";
    el.innerHTML = html;
  }

  function getActiveFilterItems() {
    var items = [];

    var searchInput = getSearchInput();
    if (searchInput && searchInput.value.trim()) {
      items.push({
        key: "search",
        label: 'Search: "' + searchInput.value.trim() + '"',
        clear: function () { searchInput.value = ""; },
      });
    }

    document.querySelectorAll(".filter-select[data-category]").forEach(function (sel) {
      if (!sel.value) return;
      var selectedLabel = sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text : sel.value;
      items.push({
        key: sel.dataset.category,
        label: formatCat(sel.dataset.category) + ": " + selectedLabel,
        clear: function () { sel.value = ""; },
      });
    });

    return items;
  }

  function renderActiveFilters() {
    var wrap = byId("active-filters");
    if (!wrap) return;
    var items = getActiveFilterItems();
    if (!items.length) {
      wrap.classList.add("hidden");
      wrap.innerHTML = "";
      return;
    }

    wrap.classList.remove("hidden");
    var html = "";
    items.forEach(function (item, idx) {
      html +=
        '<span class="filter-chip" data-chip-idx="' + idx + '">' +
        "<span>" + escapeHtml(item.label) + "</span>" +
        '<button type="button" aria-label="Remove filter">&times;</button>' +
        "</span>";
    });
    wrap.innerHTML = html;
  }

  function renderResultsSummary(data) {
    var el = byId("results-summary");
    if (!el) return;
    var showing = data.images.length;
    var total = data.total || 0;
    el.textContent =
      "Showing " + showing + " of " + total +
      " images | Page " + state.currentPage + " / " + Math.max(1, data.pages || 1) +
      " | " + state.perPage + " per page";
  }

  function loadImages(options) {
    if (!window.STEP_ID) return Promise.resolve();
    options = options || {};
    showGridSkeleton();

    return fetch("/api/images/" + window.STEP_ID + "?" + buildApiParams())
      .then(function (r) { return r.json(); })
      .then(function (data) {
        state.currentImages = data.images || [];
        state.totalPages = data.pages || 1;

        if (state.currentPage > state.totalPages) {
          state.currentPage = state.totalPages;
          return loadImages(options);
        }

        renderGrid(data);
        renderPagination(data);
        renderActiveFilters();
        renderResultsSummary(data);
        syncUrl(options.historyMode || "replace", isOverlayOpen());

        if (options.openImageFromUrl) {
          openDetailFromUrl();
        }
      })
      .catch(function (err) {
        var grid = getGrid();
        if (grid) {
          grid.classList.remove("is-loading");
          grid.innerHTML = '<div class="empty-state">Error loading images.</div>';
        }
        console.error(err);
      });
  }

  function resetDetailSections() {
    var ids = [
      "detail-labels",
      "detail-scores",
      "detail-rejection",
      "detail-soil-scores",
      "detail-resize-compare",
      "detail-dimensions",
      "detail-duplicates",
      "detail-compare-wrap",
    ];
    ids.forEach(function (id) {
      var el = byId(id);
      if (el) el.innerHTML = "";
    });
  }

  function updateDetailCounter() {
    var el = byId("detail-counter");
    if (!el) return;
    var imgNum = state.currentIndex + 1;
    var imgTotal = state.currentImages.length;
    var txt = "Image " + imgNum + " / " + imgTotal;
    if (state.totalPages > 1) {
      txt += " | Page " + state.currentPage + " / " + state.totalPages;
    }
    el.textContent = txt;
  }

  function buildDetailUrl(filename) {
    var detailUrl = "/api/image-detail/" + window.STEP_ID + "/" + encodeURIComponent(filename);
    if (window.VIEW_MODE) detailUrl += "?view=" + window.VIEW_MODE;
    return detailUrl;
  }

  var DETAIL_CACHE_MAX = 100;

  function fetchDetail(filename) {
    if (state.detailCache.has(filename)) {
      /* Move to end so recently-used entries survive eviction */
      var cached = state.detailCache.get(filename);
      state.detailCache.delete(filename);
      state.detailCache.set(filename, cached);
      return Promise.resolve(cached);
    }
    return fetch(buildDetailUrl(filename))
      .then(function (r) { return r.json(); })
      .then(function (detail) {
        state.detailCache.set(filename, detail);
        /* Evict oldest entry if cache exceeds limit */
        if (state.detailCache.size > DETAIL_CACHE_MAX) {
          var oldest = state.detailCache.keys().next().value;
          state.detailCache.delete(oldest);
        }
        return detail;
      });
  }

  function prefetchAdjacentDetails() {
    [state.currentIndex - 1, state.currentIndex + 1].forEach(function (idx) {
      var img = state.currentImages[idx];
      if (!img || state.detailCache.has(img.filename)) return;
      fetchDetail(img.filename).catch(function () { /* ignore prefetch errors */ });
    });
  }

  function setOverlayLayout() {
    var overlay = getOverlay();
    if (!overlay) return;
    var content = overlay.querySelector(".detail-content");
    if (!content) return;
    if (window.STEP_ID === "download" || window.STEP_ID === "resize") {
      content.classList.add("layout-centered");
    } else {
      content.classList.remove("layout-centered");
    }
  }

  function openDetail(index, options) {
    options = options || {};
    var img = state.currentImages[index];
    if (!img) return;
    state.currentIndex = index;
    var detailToken = Date.now() + ":" + img.filename;
    state.activeDetailToken = detailToken;

    var overlay = getOverlay();
    if (!overlay) return;
    if (!isOverlayOpen()) state.lastFocused = document.activeElement;
    overlay.classList.remove("hidden");
    document.body.style.overflow = "hidden";
    setOverlayLayout();

    var closeBtn = byId("detail-close");
    if (closeBtn) closeBtn.focus();

    var detailImg = byId("detail-img");
    if (detailImg) {
      detailImg.classList.remove("img-broken");
      detailImg.classList.remove("loaded");
      detailImg.src = img.url;
      detailImg.alt = img.filename;
      detailImg.style.display = "";
      detailImg.onload = function () { this.classList.add("loaded"); };
      detailImg.onerror = function () { handleBrokenImage(this); };
    }
    var title = byId("detail-filename");
    if (title) title.textContent = img.filename;

    resetDetailSections();
    updateDetailCounter();
    setDetailLoading(true);

    syncUrl(options.historyMode || "replace", true);

    fetchDetail(img.filename)
      .then(function (detail) {
        if (state.activeDetailToken !== detailToken || state.currentIndex !== index) return;
        renderDetail(detail);
        setDetailLoading(false);
        prefetchAdjacentDetails();
      })
      .catch(function (err) {
        console.error(err);
        setDetailLoading(false);
      });
  }

  function openDetailFromUrl() {
    var params = new URLSearchParams(window.location.search);
    var imageFromUrl = params.get("image");
    if (!imageFromUrl) {
      if (isOverlayOpen()) closeDetail();
      return;
    }

    var idx = state.currentImages.findIndex(function (img) { return img.filename === imageFromUrl; });
    if (idx >= 0) {
      openDetail(idx, { historyMode: "replace" });
    } else if (isOverlayOpen()) {
      closeDetail();
    }
  }

  function closeDetail() {
    var overlay = getOverlay();
    if (!overlay) return;
    overlay.classList.add("hidden");
    document.body.style.overflow = "";
    state.currentIndex = -1;
    state.activeDetailToken = null;
    var cmpWrap = byId("detail-compare-wrap");
    if (cmpWrap) cmpWrap.innerHTML = "";
    syncUrl("replace", false);
    if (state.lastFocused && typeof state.lastFocused.focus === "function") {
      state.lastFocused.focus();
      state.lastFocused = null;
    }
  }

  function trapOverlayFocus(evt) {
    if (evt.key !== "Tab" || !isOverlayOpen()) return;
    var overlay = getOverlay();
    if (!overlay) return;
    var focusables = overlay.querySelectorAll(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    );
    if (!focusables.length) return;
    var first = focusables[0];
    var last = focusables[focusables.length - 1];

    if (evt.shiftKey && document.activeElement === first) {
      evt.preventDefault();
      last.focus();
    } else if (!evt.shiftKey && document.activeElement === last) {
      evt.preventDefault();
      first.focus();
    }
  }

  function renderDetail(detail) {
    var detailImg = byId("detail-img");
    var resizeCompare = byId("detail-resize-compare");
    if (resizeCompare && detail.original_url) {
      if (detailImg) detailImg.style.display = "none";
      var html = '<div class="resize-comparison">';
      html += '<div class="resize-side">';
      html += '<div class="resize-label">Original';
      if (detail.original_dimensions) {
        html += ' <span class="resize-dims">' + detail.original_dimensions.width + "x" + detail.original_dimensions.height + "px</span>";
      }
      html += "</div>";
      html += '<img src="' + escapeHtml(detail.original_url) + '" alt="Original">';
      html += "</div>";
      html += '<div class="resize-arrow">-&gt;</div>';
      html += '<div class="resize-side">';
      html += '<div class="resize-label">Resized';
      if (detail.dimensions) {
        html += ' <span class="resize-dims">' + detail.dimensions.width + "x" + detail.dimensions.height + "px</span>";
      }
      html += "</div>";
      html += '<img src="' + escapeHtml(detail.url) + '" alt="Resized">';
      html += "</div></div>";
      resizeCompare.innerHTML = html;
    } else if (detailImg) {
      detailImg.style.display = "";
      if (resizeCompare) resizeCompare.innerHTML = "";
    }

    var dimsEl = byId("detail-dimensions");
    if (dimsEl && detail.dimensions) {
      dimsEl.innerHTML = '<span class="dims-badge">' + detail.dimensions.width + " x " + detail.dimensions.height + " px</span>";
    }

    var labelsEl = byId("detail-labels");
    if (labelsEl && detail.labels) {
      var labelsHtml = "";
      window.CATEGORIES.forEach(function (cat) {
        var val = detail.labels[cat] || "-";
        var conf = "";
        if (detail.scores && detail.scores[cat]) {
          conf = '<span class="detail-label-conf">(' + detail.scores[cat].confidence.toFixed(3) + ")</span>";
        }
        labelsHtml +=
          '<div class="detail-label-row">' +
          '<span class="detail-label-cat">' + formatCat(cat) + "</span>" +
          '<span><span class="detail-label-val">' + escapeHtml(val) + "</span>" + conf + "</span>" +
          "</div>";
      });
      labelsEl.innerHTML = labelsHtml;
    }

    var scoresEl = byId("detail-scores");
    if (scoresEl && detail.scores) {
      var scoresHtml = "";
      window.CATEGORIES.forEach(function (cat) {
        var catData = detail.scores[cat];
        if (!catData || !catData.all_scores) return;
        scoresHtml += '<div class="score-category"><div class="score-category-name">' + formatCat(cat) + "</div>";
        var entries = Object.entries(catData.all_scores).sort(function (a, b) { return b[1] - a[1]; });
        var maxScore = entries[0][1] || 0;
        entries.forEach(function (pair) {
          var lbl = pair[0];
          var score = pair[1];
          var pct = maxScore > 0 ? (score / maxScore * 100) : 0;
          var cls = lbl === catData.assigned ? "assigned" : "other";
          scoresHtml +=
            '<div class="score-bar-row">' +
            '<span class="score-bar-label">' + escapeHtml(lbl) + "</span>" +
            '<div class="score-bar-track"><div class="score-bar-fill ' + cls + '" style="width:' + pct + '%"></div></div>' +
            '<span class="score-bar-value">' + score.toFixed(3) + "</span>" +
            "</div>";
        });
        scoresHtml += "</div>";
      });
      scoresEl.innerHTML = scoresHtml;
    }

    var soilScoresEl = byId("detail-soil-scores");
    if (soilScoresEl && (detail.soil_scores || detail.overlay_scores || detail.filter_metrics)) {
      var soilHtml = "";
      var thresholds = detail.thresholds || {};
      var metrics = detail.filter_metrics || {};
      var soilThreshold = Number.isFinite(parseFloat(thresholds.soil_threshold))
        ? parseFloat(thresholds.soil_threshold)
        : 0.5;
      var overlayMargin = Number.isFinite(parseFloat(thresholds.overlay_margin))
        ? parseFloat(thresholds.overlay_margin)
        : 0.5;

      if (detail.source) {
        soilHtml += '<div class="soil-score-row"><strong>Source:</strong> ' + escapeHtml(detail.source) + "</div>";
      }
      if (detail.soil_scores) {
        var s = detail.soil_scores;
        var posVal = parseFloat(s.positive);
        var negVal = parseFloat(s.negative);
        var hasPos = Number.isFinite(posVal);
        var hasNeg = Number.isFinite(negVal);
        var posPct = hasPos ? Math.min(Math.max(posVal * 100, 0), 100) : 0;
        var negPct = hasNeg ? Math.min(Math.max(negVal * 100, 0), 100) : 0;
        var soilMarkerPct = Math.min(Math.max(soilThreshold * 100, 0), 100);
        var posColor = hasPos && posVal >= soilThreshold ? "var(--good)" : "var(--bad)";
        var gapVal = Number.isFinite(parseFloat(metrics.soil_gap)) ? parseFloat(metrics.soil_gap) : null;
        var marginVal = Number.isFinite(parseFloat(metrics.soil_margin)) ? parseFloat(metrics.soil_margin) : null;

        soilHtml += '<div class="threshold-section"><strong>Soil filter</strong>';
        soilHtml += '<div class="threshold-bar-row"><span class="threshold-label">Positive</span>';
        soilHtml += '<div class="threshold-track"><div class="threshold-fill threshold-good" style="width:' + posPct + '%"></div>';
        soilHtml += '<div class="threshold-marker" style="left:' + soilMarkerPct + '%" title="Soil threshold: ' + soilThreshold.toFixed(3) + '"></div></div>';
        soilHtml += '<span class="threshold-value" style="color:' + posColor + '">' + (hasPos ? posVal.toFixed(3) : "N/A") + "</span></div>";

        soilHtml += '<div class="threshold-bar-row"><span class="threshold-label">Negative</span>';
        soilHtml += '<div class="threshold-track"><div class="threshold-fill threshold-bad" style="width:' + negPct + '%"></div></div>';
        soilHtml += '<span class="threshold-value">' + (hasNeg ? negVal.toFixed(3) : "N/A") + "</span></div>";

        if (gapVal !== null) {
          soilHtml += '<div class="soil-score-row" style="font-size:.75rem;color:var(--text-muted)">positive-negative=' + gapVal.toFixed(3) + "</div>";
        }
        if (marginVal !== null) {
          soilHtml += '<div class="soil-score-row" style="font-size:.75rem;color:var(--text-muted)">positive-threshold=' + marginVal.toFixed(3) + "</div>";
        }
        soilHtml += '<div class="soil-score-row" style="font-size:.75rem;color:var(--text-muted)">rule: positive &gt;= ' + soilThreshold.toFixed(3) + " and positive &gt; negative</div>";
        soilHtml += "</div>";
      }
      if (detail.overlay_scores) {
        var o = detail.overlay_scores;
        var ovVal = parseFloat(o.overlay_score);
        var clVal = parseFloat(o.clean_score);
        var hasOv = Number.isFinite(ovVal);
        var hasCl = Number.isFinite(clVal);
        var ovPct = hasOv ? Math.min(Math.max(ovVal * 100, 0), 100) : 0;
        var clPct = hasCl ? Math.min(Math.max(clVal * 100, 0), 100) : 0;
        var deltaVal = Number.isFinite(parseFloat(metrics.overlay_delta))
          ? parseFloat(metrics.overlay_delta)
          : (hasOv && hasCl ? (ovVal - clVal) : null);
        var deltaNorm = deltaVal !== null ? Math.min(Math.max((deltaVal + 0.5) * 100, 0), 100) : 0;
        var overlayMarkerPct = Math.min(Math.max((overlayMargin + 0.5) * 100, 0), 100);
        var deltaPass = deltaVal !== null ? deltaVal <= overlayMargin : null;

        soilHtml += '<div class="threshold-section"><strong>Overlay filter</strong>';
        soilHtml += '<div class="threshold-bar-row"><span class="threshold-label">Overlay</span>';
        soilHtml += '<div class="threshold-track"><div class="threshold-fill threshold-bad" style="width:' + ovPct + '%"></div></div>';
        soilHtml += '<span class="threshold-value">' + (hasOv ? ovVal.toFixed(3) : "N/A") + "</span></div>";

        soilHtml += '<div class="threshold-bar-row"><span class="threshold-label">Clean</span>';
        soilHtml += '<div class="threshold-track"><div class="threshold-fill threshold-good" style="width:' + clPct + '%"></div></div>';
        soilHtml += '<span class="threshold-value">' + (hasCl ? clVal.toFixed(3) : "N/A") + "</span></div>";

        if (deltaVal !== null) {
          soilHtml += '<div class="threshold-bar-row"><span class="threshold-label">Delta</span>';
          soilHtml += '<div class="threshold-track"><div class="threshold-fill ' + (deltaPass ? "threshold-good" : "threshold-bad") + '" style="width:' + deltaNorm + '%"></div>';
          soilHtml += '<div class="threshold-marker" style="left:' + overlayMarkerPct + '%" title="Overlay margin: ' + overlayMargin.toFixed(3) + '"></div></div>';
          soilHtml += '<span class="threshold-value" style="color:' + (deltaPass ? "var(--good)" : "var(--bad)") + '">' + deltaVal.toFixed(3) + "</span></div>";
        }
        soilHtml += '<div class="soil-score-row" style="font-size:.75rem;color:var(--text-muted)">rule: (overlay-clean) &gt; ' + overlayMargin.toFixed(3) + " => reject</div>";
        soilHtml += '<div class="soil-score-row" style="font-size:.75rem;color:var(--text-muted)">flagged=' + escapeHtml(o.flagged) + "</div>";
        soilHtml += "</div>";
      }
      if (metrics.decision_band) {
        soilHtml += '<div class="soil-score-row" style="font-size:.75rem;color:var(--text-muted)">decision band=' + escapeHtml(metrics.decision_band) + " | nearest boundary=" + escapeHtml(metrics.nearest_boundary || "unknown") + "</div>";
      }
      soilScoresEl.innerHTML = soilHtml;
    }

    var rejectionEl = byId("detail-rejection");
    if (rejectionEl && detail.rejection) {
      var r = detail.rejection;
      var thresholdsInfo = detail.thresholds || {};
      var soilThresholdInfo = Number.isFinite(parseFloat(thresholdsInfo.soil_threshold))
        ? parseFloat(thresholdsInfo.soil_threshold).toFixed(3)
        : null;
      var overlayMarginInfo = Number.isFinite(parseFloat(thresholdsInfo.overlay_margin))
        ? parseFloat(thresholdsInfo.overlay_margin).toFixed(3)
        : null;
      var reasonLabel = r.reason === "overlay" ? "Overlay (watermark/text)" : "Low soil score";
      var reasonClass = r.reason === "overlay" ? "rejection-wm" : "rejection-score";
      var rejectionHtml =
        '<div class="rejection-info ' + reasonClass + '">' +
        '<div class="rejection-reason"><strong>Rejection reason:</strong> ' + reasonLabel + "</div>";
      if (r.positive_score) rejectionHtml += '<div class="rejection-detail">Soil positive: <strong>' + escapeHtml(r.positive_score) + "</strong></div>";
      if (r.negative_score) rejectionHtml += '<div class="rejection-detail">Soil negative: <strong>' + escapeHtml(r.negative_score) + "</strong></div>";
      if (r.overlay_score) rejectionHtml += '<div class="rejection-detail">Overlay score: <strong>' + escapeHtml(r.overlay_score) + "</strong></div>";
      if (r.clean_score) rejectionHtml += '<div class="rejection-detail">Clean score: <strong>' + escapeHtml(r.clean_score) + "</strong></div>";
      if (r.reason === "overlay" && overlayMarginInfo) {
        rejectionHtml += '<div class="rejection-detail">Rule: overlay-clean &gt; <strong>' + overlayMarginInfo + "</strong></div>";
      }
      if (r.reason !== "overlay" && soilThresholdInfo) {
        rejectionHtml += '<div class="rejection-detail">Rule: positive &ge; <strong>' + soilThresholdInfo + "</strong> and positive &gt; negative</div>";
      }
      rejectionHtml += "</div>";
      rejectionEl.innerHTML = rejectionHtml;
    } else if (rejectionEl && detail.removed) {
      var sourceLabel = detail.source || "unknown";
      var removedHtml =
        '<div class="rejection-info rejection-score">' +
        '<div class="rejection-reason"><strong>Status:</strong> Removed as duplicate</div>' +
        '<div class="rejection-detail">Source: <strong>' + escapeHtml(sourceLabel) + "</strong></div>";
      if (detail.duplicate_of) {
        removedHtml += '<div class="rejection-detail">Duplicate of: <strong>' + escapeHtml(detail.duplicate_of) + "</strong></div>";
      }
      removedHtml += "</div>";
      rejectionEl.innerHTML = removedHtml;
    }

    var dupsEl = byId("detail-duplicates");
    if (dupsEl) {
      var allDups = [];
      var groupLabel = "";

      if (detail.duplicates && detail.duplicates.length > 0) {
        allDups = detail.duplicates;
        groupLabel = "Removed Duplicates (" + allDups.length + ")";
      } else if (detail.removed && detail.duplicate_of) {
        allDups = [{ filename: detail.duplicate_of, url: detail.duplicate_of_url, kept: true }];
        if (detail.duplicate_siblings) {
          detail.duplicate_siblings.forEach(function (sibling) { allDups.push(sibling); });
        }
        groupLabel = "Duplicate Group (" + allDups.length + ")";
      }

      if (allDups.length > 0) {
        var dupsHtml = '<div class="dup-group">';
        dupsHtml += '<button class="dup-toggle-btn" type="button">';
        dupsHtml += '<span class="dup-toggle-icon">&#9654;</span> ' + groupLabel + "</button>";
        dupsHtml += '<div class="dup-thumbs">';
        allDups.forEach(function (dup) {
          var badge = dup.kept ? '<span class="dup-badge dup-badge-kept">kept</span>' : "";
          var name = dup.filename || "";
          dupsHtml += '<div class="dup-thumb" title="' + escapeHtml(name) + '">';
          dupsHtml += '<img src="' + escapeHtml(dup.url) + '" alt="' + escapeHtml(name) + '">';
          dupsHtml += badge;
          dupsHtml += '<div class="dup-thumb-name">' + escapeHtml(name.length > 20 ? (name.substring(0, 18) + "...") : name) + "</div>";
          dupsHtml += "</div>";
        });
        dupsHtml += "</div></div>";
        dupsEl.innerHTML = dupsHtml;

        var dupToggle = dupsEl.querySelector(".dup-toggle-btn");
        if (dupToggle) {
          dupToggle.addEventListener("click", function () {
            this.parentNode.classList.toggle("expanded");
          });
        }
      }
    }

    var cmpWrap = byId("detail-compare-wrap");
    if (cmpWrap && (detail.duplicate_of_url || (detail.duplicates && detail.duplicates.length > 0))) {
      var compareWith = null;
      var compareLabel = "";
      if (detail.duplicate_of_url) {
        compareWith = detail.duplicate_of_url;
        compareLabel = "Kept: " + escapeHtml(detail.duplicate_of);
      } else if (detail.duplicates && detail.duplicates.length > 0) {
        compareWith = detail.duplicates[0].url;
        compareLabel = "Removed: " + escapeHtml(detail.duplicates[0].filename);
      }
      if (compareWith) {
        var cmpHtml = '<div class="compare-section">';
        cmpHtml += '<button class="btn btn-small compare-toggle-btn" id="btn-compare-toggle" type="button">Compare side-by-side</button>';
        cmpHtml += '<div class="compare-view hidden" id="compare-view">';
        cmpHtml += '<div class="compare-side"><div class="compare-label">Current: ' + escapeHtml(detail.filename) + "</div>";
        cmpHtml += '<img src="' + escapeHtml(detail.url) + '" alt="Current"></div>';
        cmpHtml += '<div class="compare-side"><div class="compare-label">' + compareLabel + "</div>";
        cmpHtml += '<img src="' + escapeHtml(compareWith) + '" alt="Compare"></div>';
        cmpHtml += "</div></div>";
        cmpWrap.innerHTML = cmpHtml;

        var toggleBtn = byId("btn-compare-toggle");
        var compareView = byId("compare-view");
        if (toggleBtn && compareView) {
          toggleBtn.addEventListener("click", function () {
            compareView.classList.toggle("hidden");
            toggleBtn.textContent = compareView.classList.contains("hidden")
              ? "Compare side-by-side"
              : "Hide comparison";
          });
        }
      }
    }

  }

  function navigatePrev() {
    if (state.currentIndex > 0) {
      openDetail(state.currentIndex - 1, { historyMode: "replace" });
      return;
    }
    if (state.currentPage <= 1) return;

    state.currentPage -= 1;
    loadImages({ historyMode: "replace" }).then(function () {
      if (state.currentImages.length > 0) {
        openDetail(state.currentImages.length - 1, { historyMode: "replace" });
      }
    });
  }

  function navigateNext() {
    if (state.currentIndex < state.currentImages.length - 1) {
      openDetail(state.currentIndex + 1, { historyMode: "replace" });
      return;
    }
    if (state.currentPage >= state.totalPages) return;

    state.currentPage += 1;
    loadImages({ historyMode: "replace" }).then(function () {
      if (state.currentImages.length > 0) {
        openDetail(0, { historyMode: "replace" });
      }
    });
  }

  function onPopState() {
    state.handlingPopstate = true;
    applyUrlState();
    loadImages({ historyMode: "replace", openImageFromUrl: true }).finally(function () {
      state.handlingPopstate = false;
    });
  }

  function wireEvents() {
    var searchInput = getSearchInput();
    if (searchInput) {
      searchInput.addEventListener("input", function () {
        clearTimeout(state.debounceTimer);
        state.debounceTimer = setTimeout(function () {
          state.currentPage = 1;
          loadImages({ historyMode: "replace" });
        }, 300);
      });
    }

    document.querySelectorAll(".filter-select[data-category]").forEach(function (sel) {
      sel.addEventListener("change", function () {
        state.currentPage = 1;
        loadImages({ historyMode: "push" });
      });
    });

    var perPageSelect = getPerPageSelect();
    if (perPageSelect) {
      perPageSelect.addEventListener("change", function () {
        state.perPage = parsePositiveInt(perPageSelect.value, 60);
        state.currentPage = 1;
        loadImages({ historyMode: "push" });
      });
    }

    var densitySelect = getDensitySelect();
    if (densitySelect) {
      densitySelect.addEventListener("change", function () {
        state.density = densitySelect.value === "compact" ? "compact" : "comfy";
        applyDensityClass();
        syncUrl("push", isOverlayOpen());
      });
    }

    var clearBtn = byId("btn-clear-filters");
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        if (searchInput) searchInput.value = "";
        document.querySelectorAll(".filter-select[data-category]").forEach(function (sel) { sel.value = ""; });
        state.currentPage = 1;
        loadImages({ historyMode: "push" });
      });
    }

    /* Event delegation for pagination buttons */
    var paginationEl = byId("pagination");
    if (paginationEl) {
      paginationEl.addEventListener("click", function (evt) {
        var btn = evt.target.closest("button[data-page]");
        if (!btn) return;
        state.currentPage = parseInt(btn.dataset.page, 10);
        loadImages({ historyMode: "push" });
        window.scrollTo(0, 0);
      });
    }

    /* Event delegation for filter chip removal */
    var filtersWrap = byId("active-filters");
    if (filtersWrap) {
      filtersWrap.addEventListener("click", function (evt) {
        var btn = evt.target.closest("button");
        if (!btn) return;
        var chip = btn.closest(".filter-chip");
        if (!chip) return;
        var idx = parseInt(chip.dataset.chipIdx, 10);
        var items = getActiveFilterItems();
        if (!Number.isFinite(idx) || !items[idx]) return;
        items[idx].clear();
        state.currentPage = 1;
        loadImages({ historyMode: "push" });
      });
    }

    var closeBtn = byId("detail-close");
    if (closeBtn) closeBtn.addEventListener("click", closeDetail);

    var overlay = getOverlay();
    if (overlay) {
      overlay.addEventListener("click", function (evt) {
        if (evt.target === overlay) closeDetail();
      });
    }

    document.addEventListener("keydown", function (evt) {
      if (!isOverlayOpen()) return;
      if (evt.key === "Escape") {
        closeDetail();
        return;
      }
      trapOverlayFocus(evt);
      if (evt.target && (evt.target.tagName === "INPUT" || evt.target.tagName === "SELECT" || evt.target.tagName === "TEXTAREA")) {
        return;
      }
      if (evt.key === "ArrowLeft") navigatePrev();
      if (evt.key === "ArrowRight") navigateNext();
    });

    var prevBtn = byId("detail-prev");
    if (prevBtn) prevBtn.addEventListener("click", navigatePrev);
    var nextBtn = byId("detail-next");
    if (nextBtn) nextBtn.addEventListener("click", navigateNext);

    window.addEventListener("popstate", onPopState);
  }

  window.initBrowser = function () {
    applyUrlState();
    applyDensityClass();
    wireEvents();
    loadImages({ historyMode: "replace", openImageFromUrl: true });
  };
}());


/* ═══════════════════════════════════════════════════════════════════════════
   Evaluation Dashboard
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  var EVAL_COLORS = [
    "#059669", "#2563eb", "#7c3aed", "#db2777",
    "#ea580c", "#0891b2", "#d97706", "#4f46e5",
  ];

  var evalState = {
    filtered: [],
    total: 0,
    totalPages: 1,
    page: 1,
    perPage: 24,
  };

  function renderCategoryBarChart(metrics) {
    var canvas = document.getElementById("category-bar-chart");
    if (!canvas || !metrics.summary || !metrics.summary.category_ranking) return;
    var placeholder = canvas.parentNode.querySelector(".loading-placeholder");
    if (placeholder) placeholder.remove();
    canvas.hidden = false;

    var ranking = metrics.summary.category_ranking;
    var labels = ranking.map(function (r) { return formatCat(r[0]); });
    var values = ranking.map(function (r) { return +(r[1] * 100).toFixed(1); });
    var colors = values.map(function (v) {
      return v >= 75 ? "#059669" : (v >= 50 ? "#d97706" : "#dc2626");
    });

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: colors,
          borderWidth: 0,
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            min: 0, max: 100,
            ticks: {
              callback: function (v) { return v + "%"; },
              font: { size: 11 },
              color: "#6b7280",
            },
            grid: { color: "#e0e3e8" },
          },
          y: {
            ticks: { font: { size: 12, weight: "600" }, color: "#1a1d23" },
            grid: { display: false },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) { return ctx.parsed.x.toFixed(1) + "% accuracy"; },
            },
          },
        },
      },
    });
  }

  /* ── Calibration chart ───────────────────────────────────────────────── */
  function renderCalibrationChart(metrics) {
    var canvas = document.getElementById("calibration-chart");
    if (!canvas || !metrics.calibration) return;
    var categories = Object.keys(metrics.calibration);
    if (!categories.length) return;
    var placeholder = canvas.parentNode.querySelector(".loading-placeholder");
    if (placeholder) placeholder.remove();
    canvas.hidden = false;

    var datasets = [];

    categories.forEach(function (cat, i) {
      var cal = metrics.calibration[cat];
      var data = cal.accuracy.map(function (val, idx) {
        if (val === null) return null;
        return { x: idx, y: +(val * 100).toFixed(1) };
      }).filter(function (d) { return d !== null; });

      datasets.push({
        label: formatCat(cat),
        data: data,
        borderColor: EVAL_COLORS[i % EVAL_COLORS.length],
        backgroundColor: EVAL_COLORS[i % EVAL_COLORS.length] + "33",
        tension: 0.3,
        pointRadius: 4,
        pointHoverRadius: 6,
        fill: false,
      });
    });

    var binLabels = metrics.calibration[categories[0]].bins;

    new Chart(canvas, {
      type: "line",
      data: { labels: binLabels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: "Confidence Bin", font: { size: 12 } },
            ticks: { font: { size: 10 }, color: "#6b7280" },
            grid: { display: false },
          },
          y: {
            min: 0, max: 100,
            title: { display: true, text: "Accuracy %", font: { size: 12 } },
            ticks: {
              callback: function (v) { return v + "%"; },
              font: { size: 10 },
              color: "#6b7280",
            },
            grid: { color: "#e0e3e8" },
          },
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: { font: { size: 11 }, usePointStyle: true, padding: 15 },
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ": " + ctx.parsed.y.toFixed(1) + "%";
              },
            },
          },
        },
      },
    });
  }

  /* ── Confusion matrix renderer ───────────────────────────────────────── */
  function renderConfusionMatrices() {
    document.querySelectorAll(".eval-cm-wrap[data-cm]").forEach(function (wrap) {
      var cm;
      try { cm = JSON.parse(wrap.dataset.cm); } catch (e) { return; }

      var allLabels = {};
      Object.keys(cm).forEach(function (actual) {
        allLabels[actual] = true;
        Object.keys(cm[actual]).forEach(function (pred) { allLabels[pred] = true; });
      });
      var labels = Object.keys(allLabels).sort();
      if (!labels.length) return;

      /* find max value for intensity scaling */
      var maxVal = 0;
      labels.forEach(function (actual) {
        labels.forEach(function (pred) {
          var v = (cm[actual] && cm[actual][pred]) || 0;
          if (v > maxVal) maxVal = v;
        });
      });

      var html = '<table><thead><tr><th title="Actual \\ Predicted">Act \\ Pred</th>';
      labels.forEach(function (l) {
        html += "<th>" + escapeHtml(l) + "</th>";
      });
      html += "</tr></thead><tbody>";

      labels.forEach(function (actual) {
        html += "<tr><th>" + escapeHtml(actual) + "</th>";
        labels.forEach(function (pred) {
          var v = (cm[actual] && cm[actual][pred]) || 0;
          var isDiag = actual === pred;
          var intensity = maxVal > 0 ? (v / maxVal) : 0;
          var bgColor;
          if (v === 0) {
            bgColor = "transparent";
          } else if (isDiag) {
            bgColor = "rgba(5, 150, 105," + (0.1 + intensity * 0.5) + ")";
          } else {
            bgColor = "rgba(220, 38, 38," + (0.08 + intensity * 0.4) + ")";
          }
          var cls = isDiag && v > 0 ? ' class="eval-cm-diag"' : "";
          html += "<td" + cls + ' style="background:' + bgColor + '">';
          html += v > 0 ? v : "";
          html += "</td>";
        });
        html += "</tr>";
      });

      html += "</tbody></table>";
      wrap.innerHTML = html;
    });
  }

  /* ── Category expand / collapse ──────────────────────────────────────── */
  function wireCategories() {
    document.querySelectorAll(".eval-category-toggle").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var card = btn.closest(".eval-category-card");
        if (!card) return;
        var body = card.querySelector(".eval-category-body");
        if (!body) return;
        var expanded = card.classList.toggle("expanded");
        body.hidden = !expanded;
        btn.setAttribute("aria-expanded", expanded ? "true" : "false");
      });
    });
  }

  /* ── Sample browser ──────────────────────────────────────────────────── */
  function loadSamples() {
    var correctFilter = document.getElementById("eval-filter-correct");
    var categoryFilter = document.getElementById("eval-filter-category");
    var soilFilter = document.getElementById("eval-filter-soil");

    var params = [];
    var correctVal = correctFilter ? correctFilter.value : "";
    var categoryVal = categoryFilter ? categoryFilter.value : "";
    var soilVal = soilFilter ? soilFilter.value : "";

    if (correctVal) params.push("correct=" + encodeURIComponent(correctVal));
    if (categoryVal) params.push("category=" + encodeURIComponent(categoryVal));
    if (soilVal) params.push("soil=" + encodeURIComponent(soilVal));
    params.push("page=" + evalState.page);
    params.push("per_page=" + evalState.perPage);

    var url = "/api/eval/sample?" + params.join("&");

    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.error) {
          evalState.filtered = [];
          evalState.total = 0;
          evalState.totalPages = 1;
        } else {
          evalState.filtered = data.samples || [];
          evalState.total = data.total || 0;
          evalState.totalPages = data.total_pages || 1;
          evalState.page = data.page || 1;
        }
        renderSampleGrid();
        renderSamplePagination();
        renderSampleSummary();
      })
      .catch(function () {
        evalState.filtered = [];
        evalState.total = 0;
        evalState.totalPages = 1;
        renderSampleGrid();
        renderSamplePagination();
        renderSampleSummary();
      });
  }

  function applyFilters() {
    evalState.page = 1;
    loadSamples();
  }

  function renderSampleSummary() {
    var el = document.getElementById("eval-results-summary");
    if (!el) return;
    if (!evalState.total) { el.textContent = ""; return; }
    var start = (evalState.page - 1) * evalState.perPage + 1;
    var end = Math.min(evalState.page * evalState.perPage, evalState.total);
    el.textContent = "Showing " + start + "\u2013" + end + " of " + evalState.total + " samples";
  }

  function renderSampleGrid() {
    var grid = document.getElementById("eval-sample-grid");
    if (!grid) return;

    var pageItems = evalState.filtered;

    if (!pageItems.length) {
      grid.innerHTML = '<div class="empty-state">No matching samples.</div>';
      return;
    }

    var html = "";
    pageItems.forEach(function (s) {
      var isSoil = s.is_soil !== false;
      var hasLabels = isSoil && s.ground_truth && s.predicted;

      /* overall match status */
      var cardClass = "eval-sample-card";
      if (!isSoil) {
        cardClass += " eval-not-soil";
      } else if (hasLabels) {
        var allMatch = true;
        Object.keys(s.ground_truth).forEach(function (k) {
          if (s.predicted[k] !== s.ground_truth[k]) allMatch = false;
        });
        cardClass += allMatch ? " eval-match" : " eval-mismatch";
      }

      var imgUrl = "/images/eval/" + s.image.split("/").map(encodeURIComponent).join("/");
      var displayName = (s.image || "").split("/").pop();

      html += '<div class="' + cardClass + '">';
      html += '<img src="' + escapeHtml(imgUrl) + '" alt="' + escapeHtml(displayName) + '" loading="lazy" onload="this.classList.add(\'loaded\')" onerror="handleBrokenImage(this)">';
      html += '<div class="eval-sample-info">';
      html += '<div class="eval-sample-filename" title="' + escapeHtml(s.image) + '">' + escapeHtml(displayName) + "</div>";
      html += '<div class="eval-sample-labels">';

      if (!isSoil) {
        html += '<span class="eval-sample-tag eval-tag-not-soil">Not Soil</span>';
      } else if (hasLabels) {
        var categoryFilter = document.getElementById("eval-filter-category");
        var focusCat = categoryFilter ? categoryFilter.value : "";
        var catsToShow = focusCat ? [focusCat] : Object.keys(s.ground_truth).slice(0, 3);
        catsToShow.forEach(function (k) {
          var gt = s.ground_truth[k];
          var pr = s.predicted[k];
          if (gt === undefined) return;
          var match = pr === gt;
          var tagClass = match ? "eval-tag-match" : "eval-tag-mismatch";
          var label = match ? gt : (pr || "?") + " -> " + gt;
          html += '<span class="eval-sample-tag ' + tagClass + '" title="' + escapeHtml(formatCat(k)) + '">' + escapeHtml(label) + "</span>";
        });
      } else {
        html += '<span class="eval-sample-tag eval-tag-soil">Soil</span>';
      }

      html += "</div></div></div>";
    });

    grid.innerHTML = html;
  }

  function renderSamplePagination() {
    var el = document.getElementById("eval-pagination");
    if (!el) return;

    var totalPages = evalState.totalPages;
    if (totalPages <= 1) { el.innerHTML = ""; return; }

    var html = "";
    if (evalState.page > 1) {
      html += '<button class="btn btn-small" data-page="' + (evalState.page - 1) + '">&larr;</button>';
    }
    var startP = Math.max(1, evalState.page - 4);
    var endP = Math.min(totalPages, startP + 8);
    startP = Math.max(1, endP - 8);
    for (var p = startP; p <= endP; p++) {
      var active = p === evalState.page ? " active" : "";
      html += '<button class="btn btn-small' + active + '" data-page="' + p + '">' + p + "</button>";
    }
    if (evalState.page < totalPages) {
      html += '<button class="btn btn-small" data-page="' + (evalState.page + 1) + '">&rarr;</button>';
    }
    el.innerHTML = html;
  }

  function wireSampleFilters() {
    ["eval-filter-correct", "eval-filter-category", "eval-filter-soil"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.addEventListener("change", applyFilters);
    });
  }

  /* ── Init ─────────────────────────────────────────────────────────────── */
  window.initEvalDashboard = function () {
    var metrics = window.EVAL_METRICS;
    if (!metrics) return;

    renderCategoryBarChart(metrics);
    renderCalibrationChart(metrics);
    renderConfusionMatrices();
    wireCategories();
    wireSampleFilters();

    /* Event delegation for eval pagination */
    var evalPag = document.getElementById("eval-pagination");
    if (evalPag) {
      evalPag.addEventListener("click", function (evt) {
        var btn = evt.target.closest("button[data-page]");
        if (!btn) return;
        evalState.page = parseInt(btn.dataset.page, 10);
        loadSamples();
      });
    }

    loadSamples();
  };
}());
