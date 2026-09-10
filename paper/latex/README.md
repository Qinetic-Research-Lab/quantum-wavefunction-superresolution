# ML4PS 2026 submission (LaTeX)

Built with the **unmodified** official `neurips_2026.sty`. The only permitted
deviation is the footer, replaced per the ML4PS 2026 guidelines via
`\renewcommand{\@noticestring}{...}` in `main.tex`.

Build (any LaTeX toolchain; tectonic shown):

    python paper/make_figure1_compact.py     # writes outputs/analysis/figure1_compact.png
    cp outputs/analysis/figure1_compact.png paper/latex/figure1.png
    cd paper/latex && tectonic -X compile main.tex

Body is 4 pages; references begin on page 5. Submission mode is anonymous with
line numbers, as the template default.
