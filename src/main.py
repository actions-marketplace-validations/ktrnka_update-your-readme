from github import Github, Auth
from github.Repository import Repository
from github.PullRequest import PullRequest
import os
from argparse import ArgumentParser
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import openai
from pydantic import BaseModel, Field, ValidationError, model_validator, SecretStr
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel


load_dotenv()


def pull_request_to_markdown(pr: PullRequest, excluded_diff_types={"ipynb", "lock"}) -> str:
    """
    Format key information from the pull request as markdown suitable for LLM input
    """
    text = f"""
## [{pr.title}]){pr.html_url}
{pr.body or 'No description provided.'}

### Commit Messages
"""
    for commit in pr.get_commits():
        text += f"- {commit.commit.message}\n"

    for file in pr.get_files():
        patch = "Can't render patch."
        if file.patch and file.filename.split(".")[-1] not in excluded_diff_types:
            patch = file.patch
        text += f"""### {file.filename}\n{patch}\n\n"""

    return text


def gha_escape(s: str) -> str:
    """
    Escape a string for use in GitHub Actions outputs.
    Reference: https://github.com/orgs/community/discussions/26736#discussioncomment-3253165
    """
    return s.replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")


class ReadmeRecommendation(BaseModel):
    """
    Structured output for the README review task
    """

    should_update: bool = Field(
        description="Whether the README should be updated or not"
    )
    reason: str = Field(description="Reason for the recommendation")
    updated_readme: Optional[str] = Field(
        description="Updated README content, required if should_update is True, otherwise optional",
        default=None,
    )

    @model_validator(mode="after")
    def post_validation_check(self) -> "ReadmeRecommendation":
        if self.should_update and self.updated_readme is None:
            raise ValueError(
                "updated_readme must be provided if should_update is True")

        return self

    def to_github_actions_outputs(self):
        return f"""
should_update={self.should_update}
reason={gha_escape(self.reason)}
"""


# Copied from https://www.hatica.io/blog/best-practices-for-github-readme/
readme_guidelines = """
# README Guidelines

## Provide a Brief Overview of the Project
Include a brief but informative description of your project's purpose, functionality, and goals. This helps users quickly grasp the value of your project and determine if it's relevant to their needs.
Example: A user-friendly weather forecasting app that provides real-time data, daily forecasts, and weather alerts for locations worldwide.

## Installation and Setup
List Prerequisites and System Requirements

Clearly outline any prerequisites, such as software dependencies, system requirements, or environment variables, for your project. This helps users determine if they can use your project on their system and prepares them for the installation process.

Example:

```
## Prerequisites
- Node.js 14.x or higher
- Python 3.8 or higher
- Environment variables: OPENAI_API_KEY, ...
```

Step-by-Step Instructions for Installation and Setup

Provide clear, step-by-step instructions for installing and setting up your project. This helps users get started quickly and minimizes potential issues.

Example:

```
## Installation
git clone https://github.com/username/WeatherForecastApp.git

cd WeatherForecastApp

npm install

npm start
```

## Use Markdown for Formatting
Markdown is a lightweight markup language that makes it easy to format and style text. Use headers, lists, tables, and other elements to organize your README and make it visually appealing.

## Emphasize Readability and Clarity

* Large blocks of text can be challenging to read. Break down large paragraphs into smaller sections or bullet points to improve readability.
* Write using clear and concise language to ensure that your README is easily understood by users of varying technical expertise. Avoid using jargon or overly technical language without proper explanation.

## Avoid Ephemeral References
* Do not include references such as "recent changes," "recently improved," or similar time-based language. The README should be timeless, describing the current state of the project without assuming how new or old a feature is.

"""


def fill_prompt(
    readme: str, pull_request_markdown: str
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        # This triggers caching for this message AND all messages before it in the pipeline, also including any tool prompts
                        # Source: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
                        "cache_control": {"type": "ephemeral"},
                        # If we want prompt caching, this can't have any Langchain prompt variables in it
                        # Source: https://github.com/langchain-ai/langchain/discussions/25610
                        "text": f"""
You'll review a pull request and determine if the README should be updated, then suggest appropriate changes.
The README should be updated if it contains outdated information or if the pull request introduces major new features that are similar to those currently documented in the README.

When updating the README, be sure to:
* Keep the language timeless. Do not reference "recent" or "recently."
* Focus on the current state of the project features and requirements.

{readme_guidelines}

""",
                    }
                ]
            ),
            HumanMessage(
                content=f"""
# Existing README
{readme}

# Pull request changes
{pull_request_markdown}

# Task
Based on the above information, please provide a structured output indicating:
A) should_update: Should the README be updated?
B) reason: Why?
C) updated_readme: The updated README content (if applicable)
"""
            ),
        ]
    )


def get_model(model_name: str) -> BaseChatModel:
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://models.inference.ai.azure.com"

    SUPPORTED_GITHUB_MODELS = {"gpt-4o", "gpt-4o-mini",
                               "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"}
    if model_name not in SUPPORTED_GITHUB_MODELS:
        raise ValueError(f"{model_name} is not supported. If it's a non-OpenAI model, it's because we're using the AzureChatOpenAI wrapper which only supports the OpenAI models. If it's an o-series model, it's because the o-series doesn't support SystemMessage and I haven't implemented the fix yet. Supported models: {SUPPORTED_GITHUB_MODELS}")

    return AzureChatOpenAI(
        # Looks like deployment and model are the same? https://learn.microsoft.com/en-us/azure/ai-studio/ai-services/how-to/quickstart-github-models?tabs=python
        # azure_deployment=model_name,
        model=model_name,
        # This api_version supports structured output and o-series models
        api_version="2024-12-01-preview",
        # Note: This can now use a GitHub Actions token (GITHUB_TOKEN)
        api_key=SecretStr(os.environ["GITHUB_TOKEN"]),
        temperature=0.2,
        # max_tokens is output
        max_tokens=4000
    )


def get_readme(repo: Repository, pr: PullRequest, relative_readme_path: str, use_base_readme=False) -> str:
    """
    relative_readme_path: The path to the README file relative to the repository root
    """
    content_file = repo.get_contents(
        relative_readme_path, ref=pr.base.sha if use_base_readme else pr.head.sha
    )
    assert not isinstance(content_file, list), "get_readme: Expected a single file, not a list."

    return content_file.decoded_content.decode()


def review_pull_request(
    model: BaseChatModel,
    repo: Repository,
    pr: PullRequest,
    relative_readme_path: str,
    tries_remaining=1,
    use_base_readme=False,
) -> ReadmeRecommendation:
    try:
        readme_content = get_readme(
            repo, pr, relative_readme_path, use_base_readme)
        pr_content = pull_request_to_markdown(pr)

        # github provider:
        # BadRequestError: This happens on an unknown model
        # NotFoundError: This happened when trying Llama

        # The o1 error:
        # BadRequestError: Error code: 400 - {'error': {'message': "Unsupported value: 'messages[0].role' does not support 'system' with this model.", 'type': 'invalid_request_error', 'param': 'messages[0].role', 'code': 'unsupported_value'}}

        pipeline = fill_prompt(
            readme_content, pr_content
        ) | model.with_structured_output(ReadmeRecommendation)
        result = pipeline.invoke({})

        # Mainly to silence the type checker. Probably good to do anyway though
        assert isinstance(result, ReadmeRecommendation), "Expected a ReadmeRecommendation object"

        # In the Azure API, if we hit the length limit it'd be in the finish_reason:
        # if response.choices[0].finish_reason != CompletionsFinishReason.STOPPED:
        # (where CompletionsFinishReason is from azure.ai.inference.models)

        return result
    except openai.AuthenticationError as e:
        if isinstance(model, AzureChatOpenAI):
            raise ValueError(
                "Authentication error, make sure you're using a valid GitHub Actions token (GITHUB_TOKEN) or a personal access token if needed."
            ) from e
        else:
            raise e
    except ValidationError as e:
        if tries_remaining > 1:
            # BUG? If this happens, and we're piping stdout to a file to parse the output it may break Github's output parsing
            print("Validation error, trying again")
            return review_pull_request(model, repo, pr, relative_readme_path, tries_remaining - 1, use_base_readme=use_base_readme)
        else:
            raise e


def parse_pr_link(github_client: Github, url: str) -> tuple[Repository, PullRequest]:
    # TODO: Improve this code to be more robust
    repo_name = '/'.join(url.split('/')[-4:-2])
    pr_number = int(url.split('/')[-1])

    repo = github_client.get_repo(repo_name)
    pull_request = repo.get_pull(pr_number)
    return repo, pull_request


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--repository", "-r", type=str, required=True, help="Repository name"
    )
    parser.add_argument("--readme-relative", type=str,
                        required=True, help="README file")
    parser.add_argument("--readme-absolute", type=str,
                        required=True, help="README file")
    parser.add_argument("--pr", type=int, required=True,
                        help="Pull request number")

    parser.add_argument(
        "--model", type=str, default="gpt-4.1", help="GitHub Model to use"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "github"],
        help="Output format",
    )

    args = parser.parse_args()

    github_client = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]))
    repo = github_client.get_repo(args.repository)
    pr = repo.get_pull(args.pr)

    if pr.body and "NO README REVIEW" in pr.body:
        # Setup the result so the output is consistent
        result = ReadmeRecommendation(
            should_update=False, reason="'NO README REVIEW' in PR body"
        )
    else:
        model = get_model(args.model)
        result = review_pull_request(
            model, repo, pr, args.readme_relative)

        if result.should_update and result.updated_readme:
            with open(args.readme_absolute, "w") as f:
                f.write(result.updated_readme)

    if args.output_format == "github":
        # If running in Github Actions, this output formatting will set action outputs and be printed
        print(result.to_github_actions_outputs())
    else:
        print(result.model_dump_json())


if __name__ == "__main__":
    main()
