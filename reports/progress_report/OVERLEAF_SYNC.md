# Keeping local report files and Overleaf in sync

The project source is `reports/progress_report/main.tex`. Its figures and bibliography are beside it. The generated ZIP is a snapshot: rebuilding or pushing this repository to GitHub does **not** update an Overleaf project that was created by uploading the ZIP.

## Option 1: upload refreshed files

Works without a paid integration. Run `python scripts/build_progress_report.py`, extract `output/overleaf/progress_report_overleaf.zip`, and upload the changed source files/figures into the existing Overleaf project, replacing the old copies. Or upload the ZIP as a new project. Compile `main.tex` with pdfLaTeX. No experiment reruns are needed.

## Option 2: Overleaf Git integration

This is the most direct route for keeping an **existing** Overleaf project connected to local files. It requires Overleaf Cloud premium access, which may also be provided through an institution. Availability must be checked on the user's account.

1. In the Overleaf project's **Integrations → Git**, copy the Git clone URL.
2. Clone that project into a **separate local folder**, beside this research repository. Authenticate using Overleaf's supported Git credentials/token; do not put a token into a committed script or URL in this repository.
3. Before editing, pull the latest Overleaf changes into that checkout and reconcile any edits with `reports/progress_report/`. Keep the research repository's report sources as the working copy.
4. Build the local report, then copy the updated `main.tex`, `references.bib`, and needed `figures/` files into the Overleaf checkout. Do not copy research data, environments, checkpoints, or LaTeX build artifacts.
5. Review the diff, commit inside the Overleaf checkout, and push. Overleaf receives those source changes; recompile there.

This is an explicit pull/copy/commit/push workflow, not live synchronization on every save. A future helper could automate the copy step once the project URL and local checkout are selected. No Overleaf connection has been configured by the current report build.

## Option 3: Overleaf GitHub synchronization

Also a premium feature. Only the project owner can initialize the link. Overleaf supports creating a new project from an existing GitHub repository, or creating a new GitHub repository from an Overleaf project; its documentation says an existing Overleaf project cannot be directly linked to an already-existing GitHub repository.

For this project, a report-only repository would keep the Overleaf project smaller than syncing the entire research repository. Alternatively, importing this full repository requires choosing `reports/progress_report/main.tex` as the main document and checking compilation. Synchronization then uses Overleaf's GitHub push/pull controls; an ordinary local GitHub push does not by itself refresh the editor.

## Official instructions

- [Git versus GitHub integration, including premium requirements](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization)
- [Overleaf Git integration and project Git URL](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration)
- [GitHub synchronization and linking restrictions](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization)
