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

# Define target file extensions
TARGET_FILE_EXTENSIONS = ['.js', '.html', '.css']

def get_file_type(file_path: str) -> Optional[str]:
    """
    Determine the file type based on its extension.
    Returns 'js', 'html', 'css', or None if not a target file.
    """
    for ext in TARGET_FILE_EXTENSIONS:
        if file_path.lower().endswith(ext):
            return ext.lstrip('.')
    return None

def get_all_commits(repo_path: str, limit: Optional[int] = None) -> List[Any]:
    """
    Retrieve all unique commits from the given repository, excluding merge commits.
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
            # Skip merge commits (more than one parent)
            if len(commit.parents) > 1:
                logger.info(f"Skipping merge commit: {commit.hexsha}")
                continue

            if commit.hexsha not in unique_commits:
                unique_commits[commit.hexsha] = commit
                if limit and len(unique_commits) >= limit:
                    break

    commits_sorted = sorted(unique_commits.values(), key=lambda c: c.committed_datetime)
    logger.info(f"Total unique commits retrieved (excluding merges): {len(commits_sorted)}")
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
    
    # Initialize counters for each file type
    file_type_counters = {
        'js': 0,
        'html': 0,
        'css': 0
    }

    # **Initialize per-author statistics**
    authors_stats: Dict[str, Dict[str, Any]] = {}
    commit_dates: Dict[str, List[str]] = defaultdict(list)  # For commit frequency

    # **Initialize total line counters for each file type**
    total_added = defaultdict(int)
    total_removed = defaultdict(int)

    logger.info("Starting processing of commits for contribution statistics.")
    for i, commit in enumerate(all_commits):
        # Skip merge commits (handled already, but for safety)
        if len(commit.parents) > 1:
            logger.info(f"Skipping merge commit during processing: {commit.hexsha}")
            continue

        author = commit.author.name if commit.author else "Unknown"
        commit_date = commit.committed_datetime.strftime('%Y-%m')  # Year-Month format
        commit_dates[author].append(commit_date)
        logger.info(f"Processing commit {i + 1}/{len(all_commits)}: {commit.hexsha} by {author} on {commit_date}")

        # **Initialize author in authors_stats if not present**
        if author not in authors_stats:
            authors_stats[author] = {
                'total_commits': 0,
                'file_types': {
                    'js': {
                        'commits': 0,
                        'lines_added': 0,
                        'lines_removed': 0,
                        'avg_lines_added_per_commit': 0.0,
                        'avg_lines_removed_per_commit': 0.0,
                        'commit_frequency_per_month': {}
                    },
                    'html': {
                        'commits': 0,
                        'lines_added': 0,
                        'lines_removed': 0,
                        'avg_lines_added_per_commit': 0.0,
                        'avg_lines_removed_per_commit': 0.0,
                        'commit_frequency_per_month': {}
                    },
                    'css': {
                        'commits': 0,
                        'lines_added': 0,
                        'lines_removed': 0,
                        'avg_lines_added_per_commit': 0.0,
                        'avg_lines_removed_per_commit': 0.0,
                        'commit_frequency_per_month': {}
                    },
                }
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

        # Flags to check if this commit modifies any target file types
        modifies_file_types = set()

        for diff in diffs:
            # Determine the file path
            file_path = diff.b_path if diff.b_path else diff.a_path

            # Check if the file is one of the target types
            file_type = get_file_type(file_path)
            if not file_type:
                logger.info(f"Excluded file from processing: {file_path}")
                continue

            # **Mark that this commit modifies the file type**
            modifies_file_types.add(file_type)

            # Count lines added and removed
            added, removed = count_lines_in_diff(diff.diff)
            logger.info(f"File: {file_path} ({file_type}), Lines Added: {added}, Lines Removed: {removed}")

            # **Update total line counters**
            total_added[file_type] += added
            total_removed[file_type] += removed

            # **Update per-author statistics**
            authors_stats[author]['file_types'][file_type]['lines_added'] += added
            authors_stats[author]['file_types'][file_type]['lines_removed'] += removed

        # **Update commit counters for each modified file type**
        for ft in modifies_file_types:
            file_type_counters[ft] += 1
            authors_stats[author]['file_types'][ft]['commits'] += 1

    # **Calculate average lines per commit for each author and file type**
    for author, stats in authors_stats.items():
        for ft, ft_stats in stats['file_types'].items():
            commits = ft_stats['commits']
            if commits > 0:
                ft_stats['avg_lines_added_per_commit'] = round(ft_stats['lines_added'] / commits, 2)
                ft_stats['avg_lines_removed_per_commit'] = round(ft_stats['lines_removed'] / commits, 2)
            else:
                ft_stats['avg_lines_added_per_commit'] = 0.0
                ft_stats['avg_lines_removed_per_commit'] = 0.0

    # **Calculate commit frequency per author and file type**
    for author, dates in commit_dates.items():
        frequency = defaultdict(int)
        for date in dates:
            frequency[date] += 1
        for ft in TARGET_FILE_EXTENSIONS:
            ext = ft.lstrip('.')
            if ext in authors_stats[author]['file_types']:
                authors_stats[author]['file_types'][ext]['commit_frequency_per_month'] = dict(frequency)

    # **Sort authors by number of total commits (descending)**
    sorted_authors = sorted(authors_stats.items(), key=lambda x: x[1]['total_commits'], reverse=True)

    # **Output the total counts**
    logger.info(f"Total Unique Commits in Repository: {total_commits}")
    logger.info(f"Total Commits that modified .js files: {file_type_counters['js']}")
    logger.info(f"Total Commits that modified .html files: {file_type_counters['html']}")
    logger.info(f"Total Commits that modified .css files: {file_type_counters['css']}")
    logger.info(f"Total Lines Added across all .js files: {total_added['js']}")
    logger.info(f"Total Lines Removed across all .js files: {total_removed['js']}")
    logger.info(f"Total Lines Added across all .html files: {total_added['html']}")
    logger.info(f"Total Lines Removed across all .html files: {total_removed['html']}")
    logger.info(f"Total Lines Added across all .css files: {total_added['css']}")
    logger.info(f"Total Lines Removed across all .css files: {total_removed['css']}")

    # **Output the results**
    output_file_path = 'contribution_stats.json'
    try:
        # **Include per-author statistics and total counts in the JSON output**
        output_data = {
            "per_author_stats": {
                author: {
                    "total_commits": stats['total_commits'],
                    "file_types": {
                        ft: {
                            "commits": ft_stats['commits'],
                            "lines_added": ft_stats['lines_added'],
                            "lines_removed": ft_stats['lines_removed'],
                            "avg_lines_added_per_commit": ft_stats['avg_lines_added_per_commit'],
                            "avg_lines_removed_per_commit": ft_stats['avg_lines_removed_per_commit'],
                            "commit_frequency_per_month": ft_stats['commit_frequency_per_month']
                        }
                        for ft, ft_stats in stats['file_types'].items()
                    }
                }
                for author, stats in authors_stats.items()
            },
            "sorted_authors_by_total_commits": [
                {
                    "author": author,
                    "total_commits": stats['total_commits'],
                    "file_types": {
                        ft: {
                            "commits": ft_stats['commits'],
                            "lines_added": ft_stats['lines_added'],
                            "lines_removed": ft_stats['lines_removed'],
                            "avg_lines_added_per_commit": ft_stats['avg_lines_added_per_commit'],
                            "avg_lines_removed_per_commit": ft_stats['avg_lines_removed_per_commit'],
                            "commit_frequency_per_month": ft_stats['commit_frequency_per_month']
                        }
                        for ft, ft_stats in stats['file_types'].items()
                    }
                }
                for author, stats in sorted_authors
            ],
            "total_counts": {
                "total_commits": total_commits,
                "commits_modifying_files": file_type_counters,
                "lines_added": dict(total_added),
                "lines_removed": dict(total_removed)
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
        for ft, ft_stats in stats['file_types'].items():
            print(f"  {ft.upper()} Files:")
            print(f"    Commits that modified .{ft} files: {ft_stats['commits']}")
            print(f"    Lines Added (in .{ft} commits): {ft_stats['lines_added']}")
            print(f"    Lines Removed (in .{ft} commits): {ft_stats['lines_removed']}")
            print(f"    Average Lines Added per .{ft} Commit: {ft_stats['avg_lines_added_per_commit']}")
            print(f"    Average Lines Removed per .{ft} Commit: {ft_stats['avg_lines_removed_per_commit']}")
            print(f"    Commit Frequency per Month:")
            for month, count in sorted(ft_stats['commit_frequency_per_month'].items()):
                print(f"      {month}: {count} commits")

    # **Print total counts**
    print("\nOverall Repository Statistics:")
    print(f"  Total Unique Commits: {total_commits}")
    print(f"  Total Commits that modified .js files: {file_type_counters['js']}")
    print(f"  Total Commits that modified .html files: {file_type_counters['html']}")
    print(f"  Total Commits that modified .css files: {file_type_counters['css']}")
    print(f"  Total Lines Added across all .js files: {total_added['js']}")
    print(f"  Total Lines Removed across all .js files: {total_removed['js']}")
    print(f"  Total Lines Added across all .html files: {total_added['html']}")
    print(f"  Total Lines Removed across all .html files: {total_removed['html']}")
    print(f"  Total Lines Added across all .css files: {total_added['css']}")
    print(f"  Total Lines Removed across all .css files: {total_removed['css']}")

if __name__ == "__main__":
    main()
