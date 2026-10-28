# The public site

The live map, the article and the forecasters' guide are published from
this repository with GitHub Pages by `.github/workflows/cyclone_publish.yml`:
the map at the root (loading the globe kit as `globe-kit/`, its data under
`data/` from the `cps-live` branch), the article at `article/` with the
interactive 3-D phase space at `article/figures/phase_space_3d.html`, and
the guide at `forecasters/`. The pages link to each other with relative
links and to nothing else.

- https://jkrek17.github.io/cyclone/
- https://jkrek17.github.io/cyclone/article/
- https://jkrek17.github.io/cyclone/forecasters/

The workflow runs after every live export (`CPS live map data`, four times
a day) and daily collection (`CPS daily collection`, 20 UTC), on a push
that changes the pages, and by hand (Actions > Publish cyclone site > Run
workflow). It builds the site in the runner and deploys it through the
Pages deployment action with the workflow's own token, so there is no
secret to keep. The first successful run turns Pages on (Settings > Pages
then shows "GitHub Actions" as the source); if the account blocks that,
set the source to "GitHub Actions" by hand once.

The map's script, style and kit links are stamped with the commit on each
publish, so a browser holding a cached page never pairs it with stale
files. Pages serves everything with a ten minute max-age.
