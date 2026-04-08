RoMa checkpoint files (*.pth) live here.

- pip wheel/sdist: any *.pth present in this directory at build time is bundled into
  the matching-module wheel (see package-data for romatch).
- Git: *.pth is gitignored; run download.sh in this folder before local use or CI wheel builds.
