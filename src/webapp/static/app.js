/* Image browser logic for the step detail page. */

/* ─── Distribution chart ────────────────────────────────────────────────── */

(function () {
  // All-Natural earthy tones (Vyond)
  var COLOR_PALETTE = [
    '#357560', '#384727', '#F5D54A', '#CABDAF',
    '#527462', '#3A887C', '#C0C49B', '#D5DCDC',
    '#B87F45', '#9EB45B', '#7D8D87', '#CBC497',
    '#D3B73E', '#8B341F', '#89A68F', '#A3A47A',
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

    // Collect all unique value names across both kept and removed
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
      // Kept dataset
      datasets.push({
        label: val,
        data: categories.map(function (cat) { return distributions[cat][val] || 0; }),
        backgroundColor: valueColor(val),
        borderWidth: 0,
        stack: 'kept',
      });
    });

    if (removedDistributions) {
      allValues.forEach(function (val) {
        datasets.push({
          label: val + ' (removed)',
          data: categories.map(function (cat) {
            return (removedDistributions[cat] && removedDistributions[cat][val]) || 0;
          }),
          backgroundColor: valueColor(val) + '55',
          borderWidth: 1,
          borderColor: valueColor(val),
          borderDash: [3, 3],
          stack: 'removed',
        });
      });
    }

    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: categories.map(function (c) {
          var s = c.replace(/_/g, ' ');
          return s.charAt(0).toUpperCase() + s.slice(1);
        }),
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { stacked: true, ticks: { font: { size: 11 }, color: '#6b7280' }, grid: { display: false } },
          y: { stacked: true, ticks: { font: { size: 11 }, color: '#6b7280' }, grid: { color: '#e0e3e8' } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index',
            filter: function (item) { return item.parsed.y > 0; },
            callbacks: {
              label: function (ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y; },
            },
          },
        },
      },
    });
  };
}());


(function () {
  "use strict";

  let currentPage = 1;
  let totalPages = 1;
  let currentImages = [];
  let currentIndex = -1;
  let debounceTimer = null;

  /* ─── Grid loading ──────────────────────────────────────────────────── */

  function buildParams() {
    const params = new URLSearchParams();
    params.set("page", currentPage);
    params.set("per_page", 60);

    if (window.VIEW_MODE) {
      params.set("view", window.VIEW_MODE);
    }

    const search = document.getElementById("search-input");
    if (search && search.value.trim()) {
      params.set("search", search.value.trim());
    }

    document.querySelectorAll(".filter-select").forEach(function (sel) {
      if (sel.value) {
        params.set(sel.dataset.category, sel.value);
      }
    });

    return params.toString();
  }

  function loadImages() {
    if (!window.STEP_ID) return;

    const grid = document.getElementById("image-grid");
    if (!grid) return;
    grid.innerHTML = '<div class="loading">Loading images...</div>';

    fetch("/api/images/" + window.STEP_ID + "?" + buildParams())
      .then(function (r) { return r.json(); })
      .then(function (data) {
        currentImages = data.images;
        totalPages = data.pages;
        renderGrid(data);
        renderPagination(data);
      })
      .catch(function (err) {
        grid.innerHTML = '<div class="empty-state">Error loading images.</div>';
        console.error(err);
      });
  }

  function renderGrid(data) {
    var grid = document.getElementById("image-grid");

    if (!data.images.length) {
      grid.innerHTML = '<div class="empty-state">No images found.</div>';
      return;
    }

    var html = "";
    data.images.forEach(function (img, idx) {
      var corrClass = img.corrected ? " corrected" : "";
      var rejectedClass = img.rejection ? " rejected" : "";
      var removedClass = img.removed ? " rejected" : "";
      var labelsHtml = "";

      if (img.labels) {
        var tags = [];
        Object.keys(img.labels).forEach(function (cat) {
          var val = img.labels[cat];
          if (val) tags.push('<span class="label-tag">' + escapeHtml(val) + "</span>");
        });
        labelsHtml = '<div class="image-card-labels">' + tags.join("") + "</div>";
      }

      if (img.rejection) {
        var reasonText = img.rejection.reason === "overlay" ? "Overlay" : "Low score";
        var scoreText = "";
        if (img.rejection.reason === "overlay") {
          scoreText = "ov=" + escapeHtml(img.rejection.overlay_score || "");
        } else {
          scoreText = "pos=" + escapeHtml(img.rejection.positive_score || "");
        }
        var sourceTag = img.source ? '<span class="label-tag score-tag">' + escapeHtml(img.source) + '</span>' : '';
        labelsHtml =
          '<div class="image-card-labels">' +
          '<span class="label-tag rejection-tag">' + reasonText + "</span>" +
          '<span class="label-tag score-tag">' + scoreText + "</span>" +
          sourceTag +
          "</div>";
      }

      if (img.removed) {
        var sourceTag2 = img.source ? '<span class="label-tag score-tag">' + escapeHtml(img.source) + '</span>' : '';
        labelsHtml =
          '<div class="image-card-labels">' +
          '<span class="label-tag rejection-tag">Removed</span>' +
          sourceTag2 +
          "</div>";
      }

      // Source tag for kept images in filter/dedup/label
      if (!img.rejection && !img.removed && img.source) {
        var existingLabels = labelsHtml;
        var srcTag = '<span class="label-tag score-tag">' + escapeHtml(img.source) + '</span>';
        if (existingLabels) {
          // Insert source tag into existing labels div
          labelsHtml = existingLabels.replace('</div>', srcTag + '</div>');
        } else {
          labelsHtml = '<div class="image-card-labels">' + srcTag + '</div>';
        }
      }

      // Extract display name (last part of path for nested files)
      var displayName = img.filename.split("/").pop();

      html +=
        '<div class="image-card' + corrClass + rejectedClass + removedClass + '" data-idx="' + idx + '">' +
        '<img src="' + escapeHtml(img.url) + '" loading="lazy" alt="' + escapeHtml(displayName) + '">' +
        '<div class="image-card-info">' +
        '<div class="image-card-name" title="' + escapeHtml(img.filename) + '">' + escapeHtml(displayName) + "</div>" +
        labelsHtml +
        "</div></div>";
    });
    grid.innerHTML = html;

    // Click handlers
    grid.querySelectorAll(".image-card").forEach(function (card) {
      card.addEventListener("click", function () {
        var idx = parseInt(this.dataset.idx, 10);
        openDetail(idx);
      });
    });
  }

  function renderPagination(data) {
    var el = document.getElementById("pagination");
    if (!el) return;
    if (data.pages <= 1) { el.innerHTML = ""; return; }

    var html = "";

    // Previous button
    if (currentPage > 1) {
      html += '<button class="btn btn-small" data-page="' + (currentPage - 1) + '">&larr;</button>';
    }

    // Page numbers (show max 9 centered on current)
    var startPage = Math.max(1, currentPage - 4);
    var endPage = Math.min(data.pages, startPage + 8);
    startPage = Math.max(1, endPage - 8);

    for (var p = startPage; p <= endPage; p++) {
      var active = p === currentPage ? " active" : "";
      html += '<button class="btn btn-small' + active + '" data-page="' + p + '">' + p + "</button>";
    }

    // Next button
    if (currentPage < data.pages) {
      html += '<button class="btn btn-small" data-page="' + (currentPage + 1) + '">&rarr;</button>';
    }

    html += '<span class="page-info">' + data.total + " images</span>";
    el.innerHTML = html;

    el.querySelectorAll("button").forEach(function (btn) {
      btn.addEventListener("click", function () {
        currentPage = parseInt(this.dataset.page, 10);
        loadImages();
        window.scrollTo(0, 0);
      });
    });
  }

  /* ─── Detail panel ──────────────────────────────────────────────────── */

  function openDetail(idx) {
    currentIndex = idx;
    var img = currentImages[idx];
    if (!img) return;

    var overlay = document.getElementById("detail-overlay");
    overlay.classList.remove("hidden");
    document.body.style.overflow = "hidden";

    document.getElementById("detail-img").src = img.url;
    document.getElementById("detail-filename").textContent = img.filename;

    // Reset sections
    document.getElementById("detail-labels").innerHTML = "";
    document.getElementById("detail-scores").innerHTML = "";
    var rejectionEl = document.getElementById("detail-rejection");
    if (rejectionEl) rejectionEl.innerHTML = "";
    var soilScoresEl = document.getElementById("detail-soil-scores");
    if (soilScoresEl) soilScoresEl.innerHTML = "";
    var resizeCompare = document.getElementById("detail-resize-compare");
    if (resizeCompare) resizeCompare.innerHTML = "";
    var dimsEl = document.getElementById("detail-dimensions");
    if (dimsEl) dimsEl.innerHTML = "";
    var dupsEl = document.getElementById("detail-duplicates");
    if (dupsEl) dupsEl.innerHTML = "";

    var corrForm = document.getElementById("correction-form");
    if (corrForm) {
      var corrImage = document.getElementById("corr-image");
      if (corrImage) corrImage.value = img.filename;
      // Reset selects
      corrForm.querySelectorAll(".corr-select").forEach(function (sel) { sel.value = ""; });
    }
    var corrStatus = document.getElementById("corr-status");
    if (corrStatus) { corrStatus.textContent = ""; corrStatus.className = "corr-status"; }

    // Fetch full detail
    var detailUrl = "/api/image-detail/" + window.STEP_ID + "/" + encodeURIComponent(img.filename);
    if (window.VIEW_MODE) {
      detailUrl += "?view=" + window.VIEW_MODE;
    }
    fetch(detailUrl)
      .then(function (r) { return r.json(); })
      .then(function (detail) { renderDetail(detail); })
      .catch(function (err) { console.error(err); });
  }

  function renderDetail(detail) {
    // Resize before/after comparison
    var resizeCompare = document.getElementById("detail-resize-compare");
    if (resizeCompare && detail.original_url) {
      var mainImg = document.getElementById("detail-img");
      mainImg.style.display = "none";
      var html = '<div class="resize-comparison">';
      html += '<div class="resize-side">';
      html += '<div class="resize-label">Original';
      if (detail.original_dimensions) {
        html += ' <span class="resize-dims">' + detail.original_dimensions.width + '×' + detail.original_dimensions.height + 'px</span>';
      }
      html += '</div>';
      html += '<img src="' + escapeHtml(detail.original_url) + '" alt="Original">';
      html += '</div>';
      html += '<div class="resize-arrow">→</div>';
      html += '<div class="resize-side">';
      html += '<div class="resize-label">Resized';
      if (detail.dimensions) {
        html += ' <span class="resize-dims">' + detail.dimensions.width + '×' + detail.dimensions.height + 'px</span>';
      }
      html += '</div>';
      html += '<img src="' + escapeHtml(detail.url) + '" alt="Resized">';
      html += '</div>';
      html += '</div>';
      resizeCompare.innerHTML = html;
    } else if (resizeCompare && detail.dimensions) {
      var mainImg = document.getElementById("detail-img");
      mainImg.style.display = "";
      resizeCompare.innerHTML = '';
    } else {
      var mainImg = document.getElementById("detail-img");
      if (mainImg) mainImg.style.display = "";
    }

    // Pixel dimensions
    var dimsEl = document.getElementById("detail-dimensions");
    if (dimsEl && detail.dimensions) {
      dimsEl.innerHTML = '<span class="dims-badge">' + detail.dimensions.width + ' × ' + detail.dimensions.height + ' px</span>';
    }

    // Labels
    var labelsEl = document.getElementById("detail-labels");
    if (detail.labels) {
      var html = "";
      window.CATEGORIES.forEach(function (cat) {
        var val = detail.labels[cat] || "—";
        var conf = "";
        if (detail.scores && detail.scores[cat]) {
          conf = '<span class="detail-label-conf">(' + detail.scores[cat].confidence.toFixed(3) + ")</span>";
        }
        // Check if corrected
        var corrVal = "";
        if (detail.corrections && detail.corrections[cat] && detail.corrections[cat] !== val) {
          corrVal = ' <span style="color:var(--warn)">→ ' + escapeHtml(detail.corrections[cat]) + "</span>";
        }
        html +=
          '<div class="detail-label-row">' +
          '<span class="detail-label-cat">' + formatCat(cat) + "</span>" +
          '<span><span class="detail-label-val">' + escapeHtml(val) + "</span>" + conf + corrVal + "</span>" +
          "</div>";
      });
      labelsEl.innerHTML = html;
    }

    // Scores breakdown
    var scoresEl = document.getElementById("detail-scores");
    if (detail.scores) {
      var html = "";
      window.CATEGORIES.forEach(function (cat) {
        var catData = detail.scores[cat];
        if (!catData || !catData.all_scores) return;
        html += '<div class="score-category"><div class="score-category-name">' + formatCat(cat) + "</div>";
        var entries = Object.entries(catData.all_scores).sort(function (a, b) { return b[1] - a[1]; });
        var maxScore = entries[0][1];
        entries.forEach(function (pair) {
          var lbl = pair[0], score = pair[1];
          var pct = maxScore > 0 ? (score / maxScore * 100) : 0;
          var cls = lbl === catData.assigned ? "assigned" : "other";
          html +=
            '<div class="score-bar-row">' +
            '<span class="score-bar-label">' + escapeHtml(lbl) + "</span>" +
            '<div class="score-bar-track"><div class="score-bar-fill ' + cls + '" style="width:' + pct + '%"></div></div>' +
            '<span class="score-bar-value">' + score.toFixed(3) + "</span>" +
            "</div>";
        });
        html += "</div>";
      });
      scoresEl.innerHTML = html;
    }

    // Soil filter scores (for filter/rejected steps)
    var soilScoresEl = document.getElementById("detail-soil-scores");
    if (soilScoresEl && (detail.soil_scores || detail.overlay_scores)) {
      var html = "";
      if (detail.source) {
        html += '<div class="soil-score-row"><strong>Source:</strong> ' + escapeHtml(detail.source) + '</div>';
      }
      if (detail.soil_scores) {
        var s = detail.soil_scores;
        html += '<div class="soil-score-row"><strong>Soil filter:</strong> ';
        html += "positive=" + escapeHtml(s.positive) + " | negative=" + escapeHtml(s.negative);
        html += " | kept=" + escapeHtml(s.kept) + "</div>";
      }
      if (detail.overlay_scores) {
        var o = detail.overlay_scores;
        html += '<div class="soil-score-row"><strong>Overlay filter:</strong> ';
        html += "overlay=" + escapeHtml(o.overlay_score);
        html += " | clean=" + escapeHtml(o.clean_score);
        html += " | flagged=" + escapeHtml(o.flagged) + "</div>";
      }
      soilScoresEl.innerHTML = html;
    }

    // Rejection info (for rejected view in filter step)
    var rejectionEl = document.getElementById("detail-rejection");
    if (rejectionEl && detail.rejection) {
      var r = detail.rejection;
      var reasonLabel = r.reason === "overlay" ? "Overlay (watermark/text)"
        : "Low soil score";
      var reasonClass = r.reason === "overlay" ? "rejection-wm"
        : "rejection-score";
      var html =
        '<div class="rejection-info ' + reasonClass + '">' +
        '<div class="rejection-reason"><strong>Rejection reason:</strong> ' + reasonLabel + "</div>";

      if (r.positive_score) {
        html += '<div class="rejection-detail">Soil positive: <strong>' + escapeHtml(r.positive_score) + "</strong></div>";
      }
      if (r.negative_score) {
        html += '<div class="rejection-detail">Soil negative: <strong>' + escapeHtml(r.negative_score) + "</strong></div>";
      }
      if (r.overlay_score) {
        html += '<div class="rejection-detail">Overlay score: <strong>' + escapeHtml(r.overlay_score) + "</strong></div>";
      }
      if (r.clean_score) {
        html += '<div class="rejection-detail">Clean score: <strong>' + escapeHtml(r.clean_score) + "</strong></div>";
      }
      html += "</div>";
      rejectionEl.innerHTML = html;
    }
    if (rejectionEl && detail.removed) {
      var sourceLabel = detail.source || "unknown";
      var dupHtml =
        '<div class="rejection-info rejection-score">' +
        '<div class="rejection-reason"><strong>Status:</strong> Removed as duplicate</div>' +
        '<div class="rejection-detail">Source: <strong>' + escapeHtml(sourceLabel) + '</strong></div>';
      if (detail.duplicate_of) {
        dupHtml += '<div class="rejection-detail">Duplicate of: <strong>' + escapeHtml(detail.duplicate_of) + '</strong></div>';
      }
      dupHtml += "</div>";
      rejectionEl.innerHTML = dupHtml;
    }

    // Duplicate group thumbnails
    var dupsEl = document.getElementById("detail-duplicates");
    if (dupsEl) {
      var allDups = [];
      var groupLabel = "";

      if (detail.duplicates && detail.duplicates.length > 0) {
        // Kept image: show removed duplicates
        allDups = detail.duplicates;
        groupLabel = "Removed Duplicates (" + allDups.length + ")";
      } else if (detail.removed && detail.duplicate_of) {
        // Removed image: show the kept image + siblings
        allDups = [{filename: detail.duplicate_of, url: detail.duplicate_of_url, kept: true}];
        if (detail.duplicate_siblings) {
          detail.duplicate_siblings.forEach(function (s) { allDups.push(s); });
        }
        groupLabel = "Duplicate Group (" + allDups.length + ")";
      }

      if (allDups.length > 0) {
        var html = '<div class="dup-group">';
        html += '<button class="dup-toggle-btn" onclick="this.parentNode.classList.toggle(\'expanded\')">';
        html += '<span class="dup-toggle-icon">&#9654;</span> ' + groupLabel + '</button>';
        html += '<div class="dup-thumbs">';
        allDups.forEach(function (d) {
          var badge = d.kept ? '<span class="dup-badge dup-badge-kept">kept</span>' : '';
          html += '<div class="dup-thumb" title="' + escapeHtml(d.filename) + '">';
          html += '<img src="' + escapeHtml(d.url) + '" alt="' + escapeHtml(d.filename) + '">';
          html += badge;
          html += '<div class="dup-thumb-name">' + escapeHtml(d.filename.length > 20 ? d.filename.substring(0, 18) + '…' : d.filename) + '</div>';
          html += '</div>';
        });
        html += '</div></div>';
        dupsEl.innerHTML = html;
      }
    }

    // Pre-fill correction form with current labels
    if (detail.labels) {
      window.CATEGORIES.forEach(function (cat) {
        var sel = document.getElementById("corr-" + cat);
        if (sel) {
          // If there's an existing correction, show it
          if (detail.corrections && detail.corrections[cat]) {
            sel.value = detail.corrections[cat];
          }
        }
      });
    }
  }

  function closeDetail() {
    document.getElementById("detail-overlay").classList.add("hidden");
    document.body.style.overflow = "";
    currentIndex = -1;
  }

  /* ─── Corrections ───────────────────────────────────────────────────── */

  function saveCorrection(e) {
    e.preventDefault();
    var form = document.getElementById("correction-form");
    var data = {};
    data.image = document.getElementById("corr-image").value;

    var hasChange = false;
    window.CATEGORIES.forEach(function (cat) {
      var sel = document.getElementById("corr-" + cat);
      if (sel && sel.value) {
        data[cat] = sel.value;
        hasChange = true;
      }
    });

    var statusEl = document.getElementById("corr-status");
    if (!hasChange) {
      statusEl.textContent = "Select at least one label to change.";
      statusEl.className = "corr-status error";
      return;
    }

    fetch("/api/corrections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then(function (r) { return r.json(); })
      .then(function (resp) {
        if (resp.status === "saved") {
          statusEl.textContent = "Correction saved!";
          statusEl.className = "corr-status success";
          // Mark card as corrected
          if (currentIndex >= 0 && currentImages[currentIndex]) {
            currentImages[currentIndex].corrected = true;
            var card = document.querySelector('.image-card[data-idx="' + currentIndex + '"]');
            if (card) card.classList.add("corrected");
          }
        } else {
          statusEl.textContent = resp.error || "Error saving.";
          statusEl.className = "corr-status error";
        }
      })
      .catch(function () {
        statusEl.textContent = "Network error.";
        statusEl.className = "corr-status error";
      });
  }

  function revertCorrection() {
    var image = document.getElementById("corr-image").value;
    if (!image) return;

    fetch("/api/corrections/" + encodeURIComponent(image), { method: "DELETE" })
      .then(function (r) { return r.json(); })
      .then(function () {
        var statusEl = document.getElementById("corr-status");
        statusEl.textContent = "Correction reverted.";
        statusEl.className = "corr-status success";
        // Remove corrected class
        if (currentIndex >= 0 && currentImages[currentIndex]) {
          currentImages[currentIndex].corrected = false;
          var card = document.querySelector('.image-card[data-idx="' + currentIndex + '"]');
          if (card) card.classList.remove("corrected");
        }
        // Reset form
        document.getElementById("correction-form").querySelectorAll(".corr-select").forEach(
          function (sel) { sel.value = ""; }
        );
      });
  }

  /* ─── Utilities ─────────────────────────────────────────────────────── */

  function escapeHtml(s) {
    if (!s) return "";
    var div = document.createElement("div");
    div.textContent = String(s);
    return div.innerHTML;
  }

  function formatCat(cat) {
    return cat.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  /* ─── Event wiring ──────────────────────────────────────────────────── */

  window.initBrowser = function () {
    loadImages();

    // Filters
    var searchInput = document.getElementById("search-input");
    if (searchInput) {
      searchInput.addEventListener("input", function () {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(function () { currentPage = 1; loadImages(); }, 350);
      });
    }

    document.querySelectorAll(".filter-select").forEach(function (sel) {
      sel.addEventListener("change", function () { currentPage = 1; loadImages(); });
    });

    var clearBtn = document.getElementById("btn-clear-filters");
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        if (searchInput) searchInput.value = "";
        document.querySelectorAll(".filter-select").forEach(function (sel) { sel.value = ""; });
        currentPage = 1;
        loadImages();
      });
    }

    // Detail panel
    var closeBtn = document.getElementById("detail-close");
    if (closeBtn) closeBtn.addEventListener("click", closeDetail);

    var overlay = document.getElementById("detail-overlay");
    if (overlay) {
      overlay.addEventListener("click", function (e) {
        if (e.target === overlay) closeDetail();
      });
    }

    // Keyboard navigation
    document.addEventListener("keydown", function (e) {
      if (document.getElementById("detail-overlay").classList.contains("hidden")) return;
      if (e.key === "Escape") closeDetail();
      if (e.key === "ArrowLeft" && currentIndex > 0) openDetail(currentIndex - 1);
      if (e.key === "ArrowRight" && currentIndex < currentImages.length - 1) openDetail(currentIndex + 1);
    });

    // Nav buttons
    var prevBtn = document.getElementById("detail-prev");
    if (prevBtn) {
      prevBtn.addEventListener("click", function () {
        if (currentIndex > 0) openDetail(currentIndex - 1);
      });
    }
    var nextBtn = document.getElementById("detail-next");
    if (nextBtn) {
      nextBtn.addEventListener("click", function () {
        if (currentIndex < currentImages.length - 1) openDetail(currentIndex + 1);
      });
    }

    // Correction form
    var corrForm = document.getElementById("correction-form");
    if (corrForm) corrForm.addEventListener("submit", saveCorrection);

    var revertBtn = document.getElementById("btn-revert");
    if (revertBtn) revertBtn.addEventListener("click", revertCorrection);
  };
})();
