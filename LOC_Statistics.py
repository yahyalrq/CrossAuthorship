import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from git import Repo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting the Contribution Statistics Script.")

# Define the repository path (modify this path to your target repository)
REPO_PATH = 'REPO1'

# Define excluded files
EXCLUDED_FILES = {
    ".prettierrc.js",
    ".eslintrc.js",
    "babel.config.js",
    "index.js",
    "jest.config.js",
    "metro.config.js",
}

def is_js_file(file_path: str) -> bool:
    """Check if the file is a JavaScript file and not excluded."""
    return file_path.endswith('.js') and file_path not in EXCLUDED_FILES

def get_all_commits(repo_path: str, limit: Optional[int] = None) -> List[Any]:
    """
    Retrieve all unique commits from the given repository and sort them by date.
    """
    logger.info(f"Retrieving all commits from repository: {repo_path}")
    unique_commits = {}
    try:
        repo = Repo(repo_path)
        logger.info(f"Opened repository at {repo_path}")
    except Exception as e:
        logger.error(f"Failed to open repo at {repo_path}: {e}")
        return []
    
    # Iterate over all branches
    for branch in repo.branches:
        logger.info(f"Processing branch: {branch.name}")
        for commit in repo.iter_commits(branch):
            if commit.hexsha not in unique_commits:
                unique_commits[commit.hexsha] = commit
                if limit and len(unique_commits) >= limit:
                    break

    commits_sorted = sorted(unique_commits.values(), key=lambda c: c.committed_datetime)
    logger.info(f"Total unique commits retrieved: {len(commits_sorted)}")
    return commits_sorted[:limit] if limit else commits_sorted

def count_lines_in_diff(diff: bytes) -> Tuple[int, int]:
    """
    Count the number of lines added and removed in a diff.
    """
    added = 0
    removed = 0
    diff_text = diff.decode('utf-8', errors='replace').split('\n')
    for line in diff_text:
        if line.startswith('+') and not line.startswith('+++'):
            added += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1
    return added, removed

def main():
    logger.info("Main function started.")
    repo_path = REPO_PATH
    logger.info(f"Repository path: {repo_path}")

    try:
        repo = Repo(repo_path)
    except Exception as e:
        logger.error(f"Failed to open repository at {repo_path}: {e}")
        return

    all_commits = get_all_commits(repo_path)

    # **Initialize Counters**
    total_commits = len(all_commits)
    js_commits = 0  # Number of commits that modified at least one .js file

    # **Initialize per-author statistics**
    authors_stats: Dict[str, Dict[str, Any]] = {}
    commit_dates: Dict[str, List[str]] = defaultdict(list)  # For commit frequency

    # **Initialize total line counters**
    total_added = 0
    total_removed = 0

    logger.info("Starting processing of commits for contribution statistics.")
    for i, commit in enumerate(all_commits):
        author = commit.author.name if commit.author else "Unknown"
        commit_date = commit.committed_datetime.strftime('%Y-%m')  # Year-Month format
        commit_dates[author].append(commit_date)
        logger.info(f"Processing commit {i + 1}/{len(all_commits)}: {commit.hexsha} by {author} on {commit_date}")

        # **Initialize author in authors_stats if not present**
        if author not in authors_stats:
            authors_stats[author] = {
                'total_commits': 0,
                'js_commits': 0,
                'lines_added': 0,
                'lines_removed': 0,
                'avg_lines_added_per_commit': 0.0,
                'avg_lines_removed_per_commit': 0.0,
                'commit_frequency_per_month': {}
            }

        # **Increment total commits for the author**
        authors_stats[author]['total_commits'] += 1

        # Determine diffs for the commit
        if not commit.parents:
            # Initial commit, compare against empty tree
            NULL_TREE = repo.tree('4b825dc642cb6eb9a060e54bf8d69288fbee4904')
            diffs = commit.diff(NULL_TREE, create_patch=True)
        else:
            # Compare with first parent
            diffs = commit.diff(commit.parents[0], create_patch=True)

        # Flag to check if this commit modifies any .js file
        modifies_js = False

        for diff in diffs:
            # Determine the file path
            file_path = diff.b_path if diff.b_path else diff.a_path

            if not is_js_file(file_path):
                logger.info(f"Excluded file from processing: {file_path}")
                continue

            # **Mark that this commit modifies a .js file**
            modifies_js = True

            # Count lines added and removed
            added, removed = count_lines_in_diff(diff.diff)
            logger.info(f"File: {file_path}, Lines Added: {added}, Lines Removed: {removed}")

            # **Update total line counters**
            total_added += added
            total_removed += removed

            # **Update per-author statistics**
            authors_stats[author]['lines_added'] += added
            authors_stats[author]['lines_removed'] += removed

        # **Update commit counters for .js modifications**
        if modifies_js:
            js_commits += 1
            authors_stats[author]['js_commits'] += 1

    # **Calculate average lines per .js commit for each author**
    for author, stats in authors_stats.items():
        commits = stats['js_commits']
        if commits > 0:
            stats['avg_lines_added_per_commit'] = round(stats['lines_added'] / commits, 2)
            stats['avg_lines_removed_per_commit'] = round(stats['lines_removed'] / commits, 2)
        else:
            stats['avg_lines_added_per_commit'] = 0.0
            stats['avg_lines_removed_per_commit'] = 0.0

    # **Calculate commit frequency per author**
    for author, dates in commit_dates.items():
        frequency = defaultdict(int)
        for date in dates:
            frequency[date] += 1
        if author in authors_stats:
            authors_stats[author]['commit_frequency_per_month'] = dict(frequency)

    # **Sort authors by number of total commits (descending)**
    sorted_authors = sorted(authors_stats.items(), key=lambda x: x[1]['total_commits'], reverse=True)

    # **Output the total counts**
    logger.info(f"Total Unique Commits in Repository: {total_commits}")
    logger.info(f"Total Commits that modified .js files: {js_commits}")
    logger.info(f"Total Lines Added across all .js files: {total_added}")
    logger.info(f"Total Lines Removed across all .js files: {total_removed}")

    # Output the results
    output_file_path = 'contribution_stats.json'
    try:
        # **Include per-author statistics and total counts in the JSON output**
        output_data = {
            "per_author_stats": {
                author: {
                    "total_commits": stats['total_commits'],
                    "js_commits": stats['js_commits'],
                    "lines_added": stats['lines_added'],
                    "lines_removed": stats['lines_removed'],
                    "avg_lines_added_per_js_commit": stats['avg_lines_added_per_commit'],
                    "avg_lines_removed_per_js_commit": stats['avg_lines_removed_per_commit'],
                    "commit_frequency_per_month": stats['commit_frequency_per_month']
                }
                for author, stats in authors_stats.items()
            },
            "sorted_authors_by_total_commits": [
                {
                    "author": author,
                    "total_commits": stats['total_commits'],
                    "js_commits": stats['js_commits'],
                    "lines_added": stats['lines_added'],
                    "lines_removed": stats['lines_removed'],
                    "avg_lines_added_per_js_commit": stats['avg_lines_added_per_commit'],
                    "avg_lines_removed_per_js_commit": stats['avg_lines_removed_per_commit'],
                    "commit_frequency_per_month": stats['commit_frequency_per_month']
                }
                for author, stats in sorted_authors
            ],
            "total_counts": {
                "total_commits": total_commits,
                "js_commits": js_commits,
                "lines_added": total_added,
                "lines_removed": total_removed
            }
        }
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Contribution statistics completed. Results saved at {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save contribution statistics to {output_file_path}: {e}")

    # **Print per-author statistics**
    print("\nPer-Author Contribution Statistics:")

    for author, stats in sorted_authors:
        print(f"\nAuthor: {author}")
        print(f"  Total Commits: {stats['total_commits']}")
        print(f"  Commits that modified .js files: {stats['js_commits']}")
        print(f"  Lines Added (in .js commits): {stats['lines_added']}")
        print(f"  Lines Removed (in .js commits): {stats['lines_removed']}")
        print(f"  Average Lines Added per .js Commit: {stats['avg_lines_added_per_commit']}")
        print(f"  Average Lines Removed per .js Commit: {stats['avg_lines_removed_per_commit']}")
        print(f"  Commit Frequency per Month:")
        for month, count in sorted(stats['commit_frequency_per_month'].items()):
            print(f"    {month}: {count} commits")

    # **Print total counts**
    print("\nOverall Repository Statistics:")
    print(f"  Total Unique Commits: {total_commits}")
    print(f"  Total Commits that modified .js files: {js_commits}")
    print(f"  Total Lines Added across all .js files: {total_added}")
    print(f"  Total Lines Removed across all .js files: {total_removed}")

if __name__ == "__main__":
    main()
