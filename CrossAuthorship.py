import os
import json
import time
import re
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
import openai
from git import Repo
from difflib import SequenceMatcher
from collections import deque, defaultdict
import difflib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting the Authorship Scoring Script.")

# Read OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY", "")
logger.info("OpenAI API key retrieved from environment.")

if not openai.api_key:
    logger.error("OPENAI_API_KEY not found. Please set it as an environment variable.")
    exit(1)

REPO_ANDROID_PATH = 'PATH_REPO1'
REPO_IOS_PATH = 'PATH_REPO2'

MODEL_PRIMARY = "gpt-4o-mini"
MODEL_FALLBACK = "gpt-3.5-turbo"
MAX_CHARS = 10000  

CLASSIFICATION_FACTORS = {
    "Feature Addition": 1.2,
    "Bug Fix": 1.1,
    "Refactoring": 0.8,
    "Cosmetic": 1.0,
    "Duplicate Change": 0.5,  
    "Enhancement": 1.1         
}

EXCLUDED_FILES = {
    ".prettierrc.js",
    ".eslintrc.js",
    "babel.config.js",
    "index.js",
    "jest.config.js",
    "metro.config.js",
}

ast_cache: Dict[Tuple[str, str], float] = {}
semantic_cache: Dict[Tuple[str, str], float] = {}
diff_storage: Dict[str, List[Tuple[str, bytes]]] = defaultdict(list)  # **New: Store diffs per file**

def normalize_file_path(file_path: str, repo_path: str) -> str:
    logger.info(f"Normalizing file path: {file_path} in repo: {repo_path}")
    # Prefix with repo name to ensure uniqueness across repositories
    #repo_name = os.path.basename(os.path.normpath(repo_path))
    #normalized_path = f"{repo_name}/{file_path}"
    normalized_path = f"{file_path}"
    logger.info(f"Normalized file path: {normalized_path}")
    return normalized_path

def compute_authorship_weight(line_change_ratio: float, semantic_factor: float = 1.0, ast_factor: float = 1.0) -> float:
    logger.info(f"Computing authorship weight with line_change_ratio={line_change_ratio}, semantic_factor={semantic_factor}, ast_factor={ast_factor}")
    """
    Compute authorship weight based on changed lines and semantic/AST factors.
    """
    base_weight = 1.0 + 2.0 * line_change_ratio
    total_weight = base_weight * (1.5 * semantic_factor) * ast_factor
    logger.info(f"Computed base_weight={base_weight}, total_weight={total_weight}")
    return total_weight

def get_all_commits(repo_paths: List[str], limit: Optional[int] = None) -> List[dict]:
    logger.info(f"Retrieving all commits from repositories: {repo_paths}")
    """
    Retrieve all commits from the given repositories, merge them into a single list,
    and sort by date. Detect and handle first commits.
    """
    commit_list = []
    for repo_path in repo_paths:
        logger.info(f"Processing repository: {repo_path}")
        try:
            repo = Repo(repo_path)
            logger.info(f"Opened repository at {repo_path}")
        except Exception as e:
            logger.error(f"Failed to open repo at {repo_path}: {e}")
            continue

        # Iterate over branches if needed. If a single branch is known, specify it.
        for branch in repo.branches:
            logger.info(f"Processing branch: {branch.name} in repo: {repo_path}")
            for commit in repo.iter_commits(branch):
                files = []
                is_first_commit = not commit.parents  # True if no parents (first commit)
                logger.info(f"Processing commit: {commit.hexsha}, Author: {commit.author.name if commit.author else 'Unknown'}, Date: {datetime.fromtimestamp(commit.committed_date)}, First Commit: {is_first_commit}")

                if is_first_commit:
                    # First commit: include all files as "new content"
                    for blob in commit.tree.traverse():
                        if blob.path.endswith('.js') and blob.path not in EXCLUDED_FILES:  # Process only .js files
                            files.append((blob.path, None))  # No diff available
                            logger.info(f"First commit file added: {blob.path}")
                else:
                    # Non-first commit: prepare to compute diffs
                    files = None  

                commit_list.append({
                    'hash': commit.hexsha,
                    'author': commit.author.name if commit.author else "Unknown",
                    'date': datetime.fromtimestamp(commit.committed_date),
                    'repo_path': repo_path,
                    'files': files,
                    'commit': commit,
                    'first_commit': is_first_commit,  # Flag for first commit
                })

    commits_sorted = sorted(commit_list, key=lambda x: x['date'])
    logger.info(f"Total commits retrieved: {len(commits_sorted)}")
    return commits_sorted[:limit] if limit else commits_sorted

def get_file_content_from_commit(commit: Any, file_path: str) -> str:
    """Return the file content from a given commit."""
    logger.info(f"Retrieving content for file: {file_path} from commit: {commit.hexsha}")
    try:
        blob = commit.tree[file_path]
        content = blob.data_stream.read().decode('utf-8', errors='replace')
        logger.info(f"Retrieved content for file: {file_path} from commit: {commit.hexsha}")
        return content
    except KeyError:
        logger.warning(f"File {file_path} not found in commit {commit.hexsha}.")
        return ""

def analyze_ast_changes(diff_content: bytes, new_content: str, related_diffs: List[bytes], first_commit: bool = False) -> float:
    """
    Analyze AST changes.
    For first commits, treat the entire file as new code.
    Compare against related_diffs from the last three commits individually.
    Return the highest AST factor from the comparisons.
    """
    logger.info(f"Analyzing AST changes. First commit: {first_commit}")
    if first_commit:
        logger.info("Performing AST analysis for first commit.")
        old_content = ""  # No old content for first commit
        input_data = json.dumps({"old": old_content, "new": new_content})
        logger.info(f"Input data for AST analysis (first commit): {input_data}")
        try:
            result = subprocess.run(
                ["node", "parse_code.js"],
                input=input_data,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning("AST parser returned non-zero exit code for first commit.")
                return 1.0

            data = json.loads(result.stdout)
            function_changes = data.get("function_changes", 0)
            additions = data.get("additions", 0)
            removals = data.get("removals", 0)
            signature_changes = data.get("signature_changes", 0)

            logger.info(f"AST Analysis Output: {data}")

            weighted_score = (
                additions * 1.0 +
                removals * 0.25 +
                signature_changes * 0.7 +
                function_changes * 0.4
            )
            factor = 1.0 + (0.1 * weighted_score)
            factor = min(factor, 2.0)
            logger.info(f"Computed AST factor for first commit: {factor}")
            return factor

        except Exception as e:
            logger.error(f"Error in AST analysis for first commit: {e}")
            return 1.0

    if not diff_content:
        logger.info("No diff content provided for AST analysis.")
        return 1.0  # No changes to analyze

    # Extract current diff's added and removed lines
    current_patch_lines = diff_content.decode('utf-8', errors='replace').split('\n')
    current_old = "\n".join(line[1:] for line in current_patch_lines if line.startswith('-'))
    current_new = "\n".join(line[1:] for line in current_patch_lines if line.startswith('+'))
    logger.info(f"Current diff - Old lines: {len(current_old.splitlines())}, New lines: {len(current_new.splitlines())}")

    max_factor = 1.0  # Initialize with neutral factor

    for idx, related_diff in enumerate(related_diffs):
        logger.info(f"Analyzing related diff {idx + 1}/{len(related_diffs)}")
        if not related_diff:
            logger.info("Related diff is empty, skipping.")
            continue  # Skip if no related diff

        # Extract related diff's added and removed lines
        related_patch_lines = related_diff.decode('utf-8', errors='replace').split('\n')
        related_old = "\n".join(line[1:] for line in related_patch_lines if line.startswith('-'))
        related_new = "\n".join(line[1:] for line in related_patch_lines if line.startswith('+'))
        logger.info(f"Related diff {idx + 1} - Old lines: {len(related_old.splitlines())}, New lines: {len(related_new.splitlines())}")

        # Prepare input data for AST analysis: Compare current changes against related changes
        input_data = json.dumps({
            "old": related_new,
            "new": current_new,
        })
        logger.info(f"Input data for AST comparison: {input_data}")

        try:
            result = subprocess.run(
                ["node", "parse_code.js"],
                input=input_data,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning(f"AST parser failed for related diff {idx + 1}.")
                continue  # Skip this comparison if AST parser fails

            data = json.loads(result.stdout)
            function_changes = data.get("function_changes", 0)
            additions = data.get("additions", 0)
            removals = data.get("removals", 0)
            signature_changes = data.get("signature_changes", 0)

            logger.info(f"AST Analysis Output for related diff {idx + 1}: {data}")

            weighted_score = (
                additions * 1.0 +
                removals * 0.25 +
                signature_changes * 0.7 +
                function_changes * 0.4
            )
            factor = 1.0 + (0.1 * weighted_score)
            factor = min(factor, 2.0)
            logger.info(f"Computed AST factor for related diff {idx + 1}: {factor}")

            if factor > max_factor:
                logger.info(f"Updating max_factor from {max_factor} to {factor}")
                max_factor = factor  # Keep the highest factor

        except Exception as e:
            logger.error(f"Error in AST analysis comparison for related diff {idx + 1}: {e}")
            continue  # Proceed to next related diff

    logger.info(f"Final AST factor after analysis: {max_factor}")
    return max_factor

def chunk_text(text: str, max_chars: int) -> List[str]:
    logger.info(f"Chunking text into segments of max {max_chars} characters.")
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def semantic_analysis_of_change(current_diff: bytes, related_diff: bytes, first_commit: bool = False) -> float:
    """
    Classify changes as Refactoring, Feature Addition, Bug Fix, or Cosmetic using OpenAI model.
    Returns a numeric factor based on the classification.
    Only compares the current diff section against the last commit from other author.
    """
    logger.info("Starting semantic analysis of change.")
    if first_commit:
        logger.info("First commit detected. Assigning classification as 'Feature Addition'.")
        classification = "Feature Addition"  # Entire file is new
        factor = classification_to_factor(classification)
        logger.info(f"Classification: {classification}, Factor: {factor}")
        return factor

    if current_diff:
        current_patch_lines = current_diff.decode('utf-8', errors='replace').split('\n')
        current_old = "\n".join(line[1:] for line in current_patch_lines if line.startswith('-'))
        current_new = "\n".join(line[1:] for line in current_patch_lines if line.startswith('+'))
        logger.info(f"Current diff - Old lines: {len(current_old.splitlines())}, New lines: {len(current_new.splitlines())}")
    else:
        logger.info("No current diff content available for semantic analysis.")
        return 1.0  # No changes to analyze

    # Compare with related_diff (entire file content from the last commit by other authors)
    if related_diff:
        related_file_content = related_diff.decode('utf-8', errors='replace')
        logger.info("Related diff content decoded for semantic analysis.")
    else:
        related_file_content = ""
        logger.info("No related diff content available for semantic analysis.")

    # **New: Only compare changed sections against the entire related file content**
    prompt = (
        "You are a code analysis assistant."
        " Determine if the new code already existed in the related existing file content below.\n\n"
        "Related Existing File Content:\n"
        f"{related_file_content}\n\n"
        f"NEW CODE:\n{current_new}\n\n"
        "Please classify the new code into one of these categories based on its relationship to the related existing file content:\n"
        "- Duplicate Change\n"
        "- Enhancement\n"
        "- New Feature\n"
        "- Bug Fix\n"
        "- Refactoring\n"
        "- Cosmetic\n"
        "Respond with only the category name without any explanation."
    )
    logger.info(f"Prompt for semantic analysis: {prompt}")

    response = _retry_semantic_analysis(prompt, MODEL_PRIMARY)    
    model_output = response.strip()
    logger.info(f"Semantic analysis response: {model_output}")

    classification = None
    for cat in CLASSIFICATION_FACTORS.keys():
        if cat.lower() in model_output.lower():
            classification = cat
            logger.info(f"Classification matched: {classification}")
            break

    if not classification:
        classification = "Cosmetic"
        logger.info("No classification matched. Defaulting to 'Cosmetic'.")

    factor = classification_to_factor(classification)
    logger.info(f"Classification: {classification}, Factor: {factor}")
    return factor

def _retry_semantic_analysis(prompt: str, model: str) -> str:
    RETRY_LIMIT = 3
    INITIAL_BACKOFF = 5
    BACKOFF_FACTOR = 2

    attempt = 0
    backoff = INITIAL_BACKOFF
    fallback_used = (model != MODEL_PRIMARY)

    logger.info(f"Starting semantic analysis with model: {model}")
    while attempt < RETRY_LIMIT:
        try:
            logger.info(f"Semantic analysis attempt {attempt + 1} for model {model}.")
            time.sleep(1)
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            classification = completion.choices[0].message.content.strip()
            logger.info(f"Semantic analysis successful on attempt {attempt + 1}. Classification: {classification}")
            return classification

        except openai.error.RateLimitError as e:
            match = re.search(r'Please try again in (\d+(\.\d+)?)s', str(e))
            wait_time = float(match.group(1)) if match else backoff
            logger.warning(f"Rate limit hit for {model}. Waiting {wait_time}s before retrying...")
            time.sleep(wait_time)
            backoff *= BACKOFF_FACTOR
            attempt += 1

            if attempt == RETRY_LIMIT and model == MODEL_PRIMARY and not fallback_used:
                logger.info("Falling back to fallback model due to repeated rate limits.")
                return _retry_semantic_analysis(prompt, MODEL_FALLBACK)

        except Exception as e:
            logger.error(f"Error calling OpenAI API with {model}: {e}")
            return "Cosmetic"

    logger.warning(f"Exceeded retries on {model}. Returning neutral classification.")
    return "Cosmetic"

def classification_to_factor(classification: str) -> float:
    factor = CLASSIFICATION_FACTORS.get(classification, 1.0)
    logger.info(f"Mapping classification '{classification}' to factor {factor}.")
    return factor

def compute_line_change_ratio(diff_content: bytes, new_content: str, first_commit: bool = False) -> float:
    """
    Compute the ratio of changed lines.
    For first commits, treat the entire file as new code.
    """
    logger.info("Computing line change ratio.")
    if first_commit:
        total_lines = new_content.count('\n') + 1  
        logger.info(f"First commit detected. Total lines in new content: {total_lines}")
        return 1.0  # Full change

    if diff_content:
        patch_lines = diff_content.decode('utf-8', errors='replace').split('\n')
        added_lines = sum(1 for l in patch_lines if l.startswith('+') and not l.startswith('+++'))
        removed_lines = sum(1 for l in patch_lines if l.startswith('-') and not l.startswith('---'))
        total_changed = added_lines + removed_lines
        total_lines_in_diff = len([l for l in patch_lines if not l.startswith('+++') and not l.startswith('---')])
        ratio = total_changed / max(total_lines_in_diff, 1)
        logger.info(f"Added lines: {added_lines}, Removed lines: {removed_lines}, Line change ratio: {ratio}")
        return ratio

    logger.info("No diff content available. Line change ratio is 0.0.")
    return 0.0  # No changes

def jaccard_similarity(lines_a: Set[str], lines_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets of lines."""
    intersection = lines_a.intersection(lines_b)
    union = lines_a.union(lines_b)
    if not union:
        logger.info("Jaccard similarity computation has empty union. Returning 0.0.")
        return 0.0
    similarity = len(intersection) / len(union)
    logger.info(f"Jaccard similarity: {similarity}")
    return similarity

def sequence_matcher_ratio(a: str, b: str) -> float:
    """Compute a sequence-based similarity ratio using difflib."""
    ratio = SequenceMatcher(None, a, b).ratio()
    logger.info(f"SequenceMatcher ratio: {ratio}")
    return ratio

def compute_code_similarity(existing_content: str, new_content: str) -> float:
    """
    Compute similarity between two pieces of code using multiple metrics.
    """
    logger.info("Computing code similarity.")
    jaccard = jaccard_similarity(set(existing_content.splitlines()), set(new_content.splitlines()))
    seq_ratio = sequence_matcher_ratio(existing_content, new_content)
    similarity_score = (jaccard + seq_ratio) / 2
    logger.info(f"Jaccard: {jaccard}, SequenceMatcher ratio: {seq_ratio}, Similarity score: {similarity_score}")
    return similarity_score

def compute_code_similarity_factor(similarity_score: float) -> float:
    """
    Convert similarity score to a reduction factor.
    For example, higher similarity reduces the authorship weight.
    """
    # Here, if similarity is 1 (identical), reduction_factor is 0.5
    # If similarity is 0, reduction_factor is 1.0
    factor = max(1.0 - similarity_score * 0.5, 0.5)
    logger.info(f"Similarity score: {similarity_score}, Similarity factor: {factor}")
    return factor

def is_valid_commit(repo: Repo, commit_hash: str) -> bool:
    logger.info(f"Validating commit: {commit_hash} in repo: {repo.working_dir}")
    try:
        repo.commit(commit_hash)
        logger.info(f"Commit {commit_hash} is valid.")
        return True
    except Exception as e:
        logger.error(f"Invalid commit {commit_hash}: {e}")
        return False

def manually_compute_diff(repo_current: Repo, commit_current: Any, repo_previous: Repo, commit_previous: Any) -> List[Tuple[str, bytes]]:
    """
    Compute file-level differences manually between two commits, including those across repositories.
    Only process .js files not in EXCLUDED_FILES.
    Returns a list of tuples (file_path, unified_diff_bytes).
    """
    logger.info("Manually computing diffs between commits.")
    
    # Filter relevant .js files
    def get_relevant_files(commit):
        files = {blob.path for blob in commit.tree.traverse() 
                 if blob.path.endswith('.js') and blob.path not in EXCLUDED_FILES}
        logger.info(f"Relevant files in commit {commit.hexsha}: {files}")
        return files

    current_files = get_relevant_files(commit_current)
    previous_files = get_relevant_files(commit_previous)

    # Intersection of files present in both commits
    common_files = current_files & previous_files
    logger.info(f"Common files between commits {commit_previous.hexsha} and {commit_current.hexsha}: {common_files}")

    diffs = []
    for file_path in common_files:
        try:
            old_content = get_file_content_from_commit(commit_previous, file_path)
            new_content = get_file_content_from_commit(commit_current, file_path)
            if old_content != new_content:
                # Create a unified diff string as bytes
                unified_diff = ''.join(difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile='a/' + file_path,
                    tofile='b/' + file_path,
                    lineterm=''
                )).encode('utf-8')
                diffs.append((file_path, unified_diff))
                logger.info(f"Diff computed for file: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    logger.info(f"Total diffs computed manually: {len(diffs)}")
    return diffs

def check_repo_integrity(repo_path: str):
    """
    Run a repository integrity check to ensure no missing or corrupted objects.
    """
    logger.info(f"Checking repository integrity for: {repo_path}")
    try:
        repo = Repo(repo_path)
        repo.git.fsck()
        logger.info(f"Repository at {repo_path} passed integrity checks.")
    except Exception as e:
        logger.error(f"Repository integrity check failed for {repo_path}: {e}")

def main():
    logger.info("Main function started.")
    # Paths to repositories
    repo_paths = [REPO_IOS_PATH, REPO_ANDROID_PATH]
    logger.info(f"Repository paths: {repo_paths}")
    all_commits = get_all_commits(repo_paths)  # Assume get_all_commits is implemented

    file_authorship_scores: Dict[str, Dict[str, float]] = {}
    project_authorship_scores: Dict[str, float] = {}

    # Initialize a dictionary to keep track of recent commits per file
    # Key: normalized file_path, Value: deque of last 3 commits by other authors
    file_recent_commits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

    # Validate repository integrity
    for repo_path in repo_paths:
        check_repo_integrity(repo_path)

    logger.info("Starting processing of commits for authorship scoring.")
    for i, commit_data in enumerate(all_commits):
        logger.info(f"Processing commit {i + 1}/{len(all_commits)}: {commit_data['hash']} by {commit_data['author']}")
        current_commit = commit_data['commit']
        try:
            current_repo = Repo(commit_data['repo_path'])
        except Exception as e:
            logger.error(f"Failed to open repository at {commit_data['repo_path']}: {e}")
            continue
        current_author = commit_data['author']
        is_first_commit = commit_data['first_commit']
        normalized_repo_path = current_repo.working_dir

        if not is_valid_commit(current_repo, current_commit.hexsha):
            logger.error(f"Skipping invalid commit: {current_commit.hexsha}")
            continue

        if is_first_commit or i == 0:
            logger.info(f"Handling first commit: {current_commit.hexsha}")
            for blob in current_commit.tree.traverse():
                if blob.path.endswith('.js') and blob.path not in EXCLUDED_FILES:
                    new_content = blob.data_stream.read().decode('utf-8', errors='replace')
                    normalized_file_path = normalize_file_path(blob.path, current_repo.working_dir)
                    contribution_weight = 1.0  # Assign full weight for new content
                    file_scores = file_authorship_scores.setdefault(normalized_file_path, {})
                    project_authorship_scores[current_author] = project_authorship_scores.get(current_author, 0) + contribution_weight
                    file_scores[current_author] = file_scores.get(current_author, 0) + contribution_weight
                    logger.info(f"Assigned weight {contribution_weight} to author {current_author} for file {normalized_file_path}")

                    # **Save the diff for the first commit as None (no previous diff)**
                    diff_storage[normalized_file_path].append((current_commit.hexsha, None))
                    logger.info(f"Stored diff as None for first commit of file {normalized_file_path}")

                    # Add to recent commits
                    file_recent_commits[normalized_file_path].append({
                        'author': current_author,
                        'diff': None,  # No diff for first commit
                        'content': new_content
                    })
                    logger.info(f"Added first commit to recent commits for file {normalized_file_path}")
            continue

        # Handle diffs with the previous commit
        previous_commit_data = all_commits[i - 1]
        previous_commit = previous_commit_data['commit']
        try:
            previous_repo = Repo(previous_commit_data['repo_path'])
        except Exception as e:
            logger.error(f"Failed to open repository at {previous_commit_data['repo_path']}: {e}")
            continue
        normalized_previous_repo_path = previous_repo.working_dir

        # Validate previous commit
        if not is_valid_commit(previous_repo, previous_commit.hexsha):
            logger.error(f"Skipping diff for invalid commit: {previous_commit.hexsha}")
            continue

        try:
            if current_repo != previous_repo:
                logger.info(f"Repositories differ. Manually computing diffs between {previous_repo.working_dir} and {current_repo.working_dir}")
                diffs = manually_compute_diff(current_repo, current_commit, previous_repo, previous_commit)
            else:
                logger.info(f"Computing diffs within the same repository: {current_repo.working_dir}")
                diffs = current_repo.commit(current_commit.hexsha).diff(previous_commit.hexsha, create_patch=True)
                # Convert Diff objects to a uniform tuple format
                diffs = [(diff.b_path or diff.a_path, diff.diff) for diff in diffs]
                logger.info(f"Computed {len(diffs)} diffs in the same repository.")

            # **Save diffs for the current commit**
            for file_path, diff_content in diffs:
                if file_path in EXCLUDED_FILES or not file_path.endswith('.js'):
                    logger.info(f"Excluded file from diff processing: {file_path}")
                    continue

                # **Store the diff for later use**
                normalized_file_path = normalize_file_path(file_path, current_repo.working_dir)
                diff_storage[normalized_file_path].append((current_commit.hexsha, diff_content))
                logger.info(f"Stored diff for file {normalized_file_path} in commit {current_commit.hexsha}")

            for diff in diffs:
                if isinstance(diff, tuple):
                    file_path, diff_content = diff
                else:
                    # In case diffs are not tuples, though this should not happen
                    file_path = diff.b_path or diff.a_path
                    diff_content = diff.diff

                if file_path in EXCLUDED_FILES or not file_path.endswith('.js'):
                    logger.info(f"Excluded file from analysis: {file_path}")
                    continue

                new_content = get_file_content_from_commit(current_commit, file_path)
                old_content = get_file_content_from_commit(previous_commit, file_path)

                if not new_content:
                    logger.info(f"No new content for file {file_path} in commit {current_commit.hexsha}. Skipping.")
                    continue

                normalized_file_path = normalize_file_path(file_path, current_repo.working_dir)

                # **Retrieve the last three diffs from other authors for AST analysis**
                recent_commits = file_recent_commits[normalized_file_path]
                logger.info(f"Retrieved {len(recent_commits)} recent commits for AST analysis for file {normalized_file_path}")

                related_diffs=[]
                for commit in recent_commits:
                    if commit['author'] != current_author:
                        related_diffs.append(commit["diff"])
               
                logger.info(f"Retrieved {len(related_diffs)} related diffs for AST analysis for file {normalized_file_path}")

                line_change_ratio = compute_line_change_ratio(diff_content, new_content, is_first_commit)
                logger.info(f"Line change ratio for file {normalized_file_path}: {line_change_ratio}")
                ast_factor = analyze_ast_changes(diff_content, new_content, related_diffs, is_first_commit)
                logger.info(f"AST factor for file {normalized_file_path}: {ast_factor}")
                
                # **Retrieve the last commit from other authors for semantic analysis**
                last_other_commit = next((commit for commit in reversed(recent_commits) if commit['author'] != current_author), None)
                related_diff = last_other_commit['diff'] if last_other_commit else None

                semantic_factor = semantic_analysis_of_change(diff_content, related_diff, is_first_commit)
                logger.info(f"Semantic factor for file {normalized_file_path}: {semantic_factor}")
                contribution_weight = compute_authorship_weight(line_change_ratio, semantic_factor, ast_factor=ast_factor)
                logger.info(f"Contribution weight for author {current_author} on file {normalized_file_path}: {contribution_weight}")

                # ------------------ Modified Similarity Check Logic ------------------ #

                # **Similarity is now based only on changed sections against the last three diffs by other authors**
                similarity_scores = []

                for related_diff_idx, related_diff in enumerate(related_diffs, 1):
                    if related_diff:
                        past_patch_lines = related_diff.decode('utf-8', errors='replace').split('\n')
                        past_changed_code = "\n".join(line[1:] for line in past_patch_lines if line.startswith(('+', '-')))
                        logger.info(f"Related diff {related_diff_idx} changed lines extracted.")

                        current_patch_lines = diff_content.decode('utf-8', errors='replace').split('\n')
                        current_changed_code = "\n".join(line[1:] for line in current_patch_lines if line.startswith(('+', '-')))
                        logger.info("Current changed code extracted for similarity computation.")

                        similarity = compute_code_similarity(past_changed_code, current_changed_code)
                        similarity_scores.append(similarity)
                        logger.info(f"Similarity score with related diff {related_diff_idx}: {similarity}")

                if similarity_scores:
                    # For example, take the maximum similarity score
                    max_similarity = max(similarity_scores)
                    similarity_factor = compute_code_similarity_factor(max_similarity)
                    contribution_weight *= similarity_factor
                    logger.info(f"Max similarity score: {max_similarity}, Similarity factor: {similarity_factor}, Updated contribution weight: {contribution_weight}")
                else:
                    similarity_factor = 1.0  # No similar past diffs found
                    logger.info("No similarity scores computed. Similarity factor set to 1.0")

                # Update scores
                file_scores = file_authorship_scores.setdefault(normalized_file_path, {})
                project_authorship_scores[current_author] = project_authorship_scores.get(current_author, 0) + contribution_weight
                file_scores[current_author] = file_scores.get(current_author, 0) + contribution_weight
                logger.info(f"Updated authorship scores for author {current_author} on file {normalized_file_path}")

                # **Add the current commit to recent commits**
                file_recent_commits[normalized_file_path].append({
                    'author': current_author,
                    'diff': diff_content,
                    'content': new_content
                })
                logger.info(f"Added current commit to recent commits for file {normalized_file_path}")

                # Debug logging
                logger.info(f"File: {normalized_file_path}, Author: {current_author}, Similarity Factor: {similarity_factor:.2f}")

        except Exception as e:
            logger.error(f"Error computing diff between {previous_commit.hexsha} and {current_commit.hexsha}: {e}")

    # Save results
    output_file_path = 'authorship_scores.txt'
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("File Authorship Scores:\n")
            for file_path, scores in file_authorship_scores.items():
                f.write(f"{file_path}: {json.dumps(scores)}\n")
            f.write("\nProject Authorship Scores:\n")
            for author, score in project_authorship_scores.items():
                f.write(f"{author}: {score}\n")
        logger.info(f"Authorship scoring completed. Results saved at {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save authorship scores to {output_file_path}: {e}")

if __name__ == "__main__":
    main()
