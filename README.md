# Scientific_Computing_1

# Git Guide: Best Practices and Workflow

## Best Practices for Working on This Repository
- **Always create a new branch and pull request** – Do not push directly to the main branch.
- **Task-based branching** – Create a new branch for each task and label it accordingly (e.g., `adding_of_age_parameter`). Avoid using personal names as branch names.
- **Limit concurrent work** – Work on a maximum of one or two tasks at a time. Break larger tasks into smaller sub-tasks.
- **Frequent merging** – Merge small, well-tested changes as soon as possible.
- **Clean up branches** – Delete branches after they are merged. Avoid reusing the same branch name for new tasks.
- **Code reviews** – Every pull request must be reviewed by at least one person before merging.
- **Documentation** – Every function should have at least a one-line docstring before merging to the main branch.
- **Frequent commits** – Commit often to track progress effectively.
- **Avoid clutter** – Remove large commented-out code before merging. Git history retains all previous versions if needed.

---

## Git Workflow Cheat Sheet

If you don´t use GitHub Desktop (I recommend to use that), you can do it in the terminal:

### Setting Up Your Workspace
1. Make sure you are on the main branch and pull the latest changes:
   ```bash
   git checkout main
   git pull
   ```
2. Create a new branch for your task:
   ```bash
   git checkout -b "my_new_task"
   ```

### Making and Committing Changes
- Whenever you complete a small piece of code, commit it. 
- Short but descriptive commit messages are preferred, but frequent commits are more important.

To commit all changes:
```bash
git add .
git commit -m "This is my great commit message"
```

To commit a specific file:
```bash
git add /path/to/my/file_to_commit
git commit -m "Describe the change"
```

Push your changes to the remote repository:
```bash
git push
```

### Creating a Pull Request
- Once your work is ready, ensure the code is clean and well-tested.
- Push all changes and navigate to GitHub to create a pull request.
- Assign all reviewers.

### Merging and Cleaning Up
- After a pull request is approved, merge it.
- Delete the branch from GitHub and also remove it locally to keep your workspace clean:
  ```bash
  git checkout main
  git pull
  git branch -d "my_new_task"
  ```
