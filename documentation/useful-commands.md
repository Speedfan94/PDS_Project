# Git Behaviour
## Branch-Naming
feat/<SHORT_FEATURE_NAME>
## 1. Useful commands
| Functionality | Git command |
|---------------|-------------|
| Create a new branch | ```git checkout -b <BRANCH_NAME>``` |
| Check if there are updates online | ```git fetch --all``` |
| See the last commits on current branch | ```git log``` |
| Show changes of a specific commit | ```git show <COMMIT_HASH>``` |
| Review uncommitted code changes | ```git diff``` |
| See what files you have changed, which are already added to be committed etc. | ```git status``` |
| List all existing local branches | ```git branch -l``` |
| List all existing remote branches | ```git branch -r``` |
| Add files to a commit | ```git add <PATH_TO_FILE>``` |
| Push all commits to upstream/online repository | ```git push``` |
| Push new branch to upstream/online repository | ```git push -u origin <BRANCH_NAME>``` |
## 2. Daily Git Routines
### 2.1. (Morning routine) Get Updates:
1. ```git fetch --all``` - Check if there are updates
2. ```git checkout dev``` - If there are changes in dev, switch to dev branch
3. ```git pull``` - Pull the changes
4. ```git checkout <BRANCH_NAME>``` - Switch back to your feature branch
5. ```git merge dev``` - Merge the changes into your branch

### 2.2. (Evening routine) Push Changes:
1. ```git status``` - Check what files you changed
2. ```git diff``` - (Optional, but recommended:) Review your changes
3. ```git add <FILEPATHES_TO_ADD>``` - Add files you want to commit.
    Recommended to split by root folders
    (commit separately, e.g. "documentation" / "data" / "nextbike/io")
4. ```git commit -m "<COMMIT_MESSAGE>"``` - Commit your changes.
   See "commit-style.md" for detailed info about committing style
5. ```git push``` - Push changes to origin repository

## 3. Handling Merge Conflicts
If you have merge conflicts after a ```git merge``` command, do the following:
1. Go into PyCharm
2. Rightclick on any file
3. Navigate to: Git > Resolve conflicts
4. Doubleclick on listed files (these are the files including merge conflicts)
5. You enter a 3-split-screen
6. One of these screens (either left or right) is the current file in the dev branch, one is the same file in your branch & the middle screen is the resulting file. Clear all conflicts by accepting either the left or right solution or merge conflicting snippets logically.
7. Go back to console and continue your git workflow

## 4. Useful PyCharm shortcuts
| Feature | Shortcut Windows | Shortcut Mac |
|---------|------------------|--------------|
| Open a file or search for a PyCharm setting/action/menu | Double Shift | Double Shift |
| Open recent files | Strg + E | Cmd + E |
| Search in current file | Strg + F | Cmd + F |
| Search in whole project | Strg + Shift + F | Cmd + Shift + F |
| Rename variables | Shift + F6 | Shift + F6 |
| Move a code line up/down | Shift + Alt + Arrow up/down | Shift + (Alt or Cmd) + Arrow up/down |
| Delete a line | Shift + Entf | Alt + Backspace |
| Copy a whole line (without marking anything) | Strg + C | Cmd + C |
| Open/Close project/directory structure | Alt + 1 | Cmd + 1 |
