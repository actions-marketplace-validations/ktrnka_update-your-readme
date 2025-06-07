# Update Your README

This project automatically updates README files based on changes in pull requests using the GitHub API and language models.

## Features

- Suggests README updates based on the pull request description, code changes in the PR, and commit messages
- Automatically closes stale README update PRs
- Uses GitHub Models for intelligent suggestions
- Option to skip README checks for testing purposes

## Usage

### Prerequisites

- GitHub repository
- GitHub Actions Token (`GITHUB_TOKEN`)

To use this action in your GitHub workflow, add the following step to your `.github/workflows/your-workflow.yml` file, replacing the version as needed:

```yaml
- uses: ktrnka/update-your-readme@VERSION
  with:
    model: gpt-4.1  # Specify your preferred GitHub model
    readme-file: README.md
    debug: "true"
```

See `.github/workflows/suggest_readme_updates.yml` for an example.

Make sure to use the default `GITHUB_TOKEN` provided by GitHub Actions. Note: The Action will not work on PRs from forks because these secrets aren't available on workflows for those PRs.

### Model Configuration

You can specify which GitHub model to use through the `model` input parameter. Supported models include:
- `gpt-4.1` (recommended)
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-4o`
- `gpt-4o-mini`

In your repo settings, under Actions > General > Workflow Permissions, be sure to check "Allow GitHub Actions to create and approve pull requests" and allow read/write from GitHub Actions:
![Workflow Permissions](workflow_permissions.png)

### Installation and Setup

To set up the project, you need to install the 'uv' package for managing the Python environment:

1. Install uv:
   ```
   make install-uv
   ```

2. Install dependencies:
   ```
   make install
   ```

### Skipping README Check

To skip the README check for testing purposes, include "NO README REVIEW" in the pull request body. This will cause the action to exit without performing any updates.

## Development

### Testing

This project uses pytest for testing. To run the tests, execute the following command:

```
make test
```

### Code Formatting

We use Black for code formatting. To format your code, run:

```
black .
```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Ensure your code follows the project's coding standards (use Black for formatting).
2. Update the README if necessary.

### License

[MIT License](https://opensource.org/licenses/MIT)

### GitHub Actions Integration

This project includes GitHub Actions workflows that enhance the README update process:

1. **Suggest README Updates**: Defined in `.github/workflows/suggest_readme_updates.yml`, this workflow:
   - Uses the `ktrnka/update-your-readme` action
   - Runs the README update process
   - Creates a new pull request with the suggested changes
   - Adds a comment to the original pull request with a link to the suggested changes

2. **Close Stale PRs**: Defined in `.github/workflows/close_stale_prs.yml`, this workflow:
   - Runs on a daily schedule
   - Uses the `actions/stale@v9` action to automatically close stale pull requests
   - Configurable stale and close timeframes
   - Targets PRs with the "update-your-readme" label

To use these features, ensure that your repository uses the default `GITHUB_TOKEN` and passes it as a parameter to the action.

### Debugging

The action supports a `debug` input, which can be set to "true" to enable additional debugging information. This can be helpful when troubleshooting issues with the action.
