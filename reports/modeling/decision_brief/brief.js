(function () {
  // One tooltip for every chart; it enhances and never gates (each chart has a data table).
  var tip = document.createElement("div");
  tip.className = "tooltip";
  tip.hidden = true;
  tip.setAttribute("role", "status");
  document.body.appendChild(tip);

  function place(x, y) {
    var pad = 14;
    var box = tip.getBoundingClientRect();
    var left = x + pad;
    var top = y + pad;
    if (left + box.width > window.innerWidth - 8) left = x - box.width - pad;
    if (top + box.height > window.innerHeight - 8) top = y - box.height - pad;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }

  function cross(el, visible) {
    var svg = el.ownerSVGElement;
    var line = svg && svg.querySelector(".cross");
    if (!line || !el.dataset.cx) return;
    if (!visible) { line.setAttribute("hidden", ""); return; }
    line.setAttribute("x1", el.dataset.cx);
    line.setAttribute("x2", el.dataset.cx);
    line.setAttribute("y1", el.dataset.cy1);
    line.setAttribute("y2", el.dataset.cy2);
    line.removeAttribute("hidden");
  }

  function show(el, x, y) {
    tip.replaceChildren();
    el.dataset.tip.split("|").forEach(function (text, i) {
      var row = document.createElement("div");
      if (i === 0) row.className = "tt-head";
      row.textContent = text;
      tip.appendChild(row);
    });
    tip.hidden = false;
    place(x, y);
    cross(el, true);
  }

  function hide(el) {
    tip.hidden = true;
    cross(el, false);
  }

  document.querySelectorAll("[data-tip]").forEach(function (el) {
    el.addEventListener("pointerenter", function (e) { show(el, e.clientX, e.clientY); });
    el.addEventListener("pointermove", function (e) { place(e.clientX, e.clientY); });
    el.addEventListener("pointerleave", function () { hide(el); });
  });
})();
