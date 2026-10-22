// Plotly's `responsive` config only re-fits a figure on a *window* resize. In a CSS-grid theme the
// content column gets its final width after the plot's first draw, so a figure can render too wide and
// never correct until the window happens to resize (browser-dependent; Firefox keeps the wide draw).
// Observe each figure's container and resize the plot to match, so it fits in every browser.
window.addEventListener("load", function () {
    if (!window.Plotly || !window.ResizeObserver) return;
    document.querySelectorAll(".plotly-graph-div").forEach(function (gd) {
        const fit = function () {
            try {
                window.Plotly.Plots.resize(gd);
            } catch (e) {}
        };
        new ResizeObserver(fit).observe(gd.parentElement);
        fit();
    });
});
