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
| Add files to a commit | ```git add <PATH_TO_FILE>``` |
| Push all commits to upstream/online repository | ```git push``` |
| Push new branch to upstream/online repository | ```git push -u origin <BRANCH_NAME>``` |
## 2. Daily Git Routines
### 2.1. (Morning) Get Updates:
1. Check if there are updates: ```git fetch --all```
2. If there are changes in dev, switch to dev branch: ```git checkout dev```
3. Pull the changes: ```git pull```
4. Switch back to your feature branch: ```git checkout <BRANCH_NAME>```
5. Merge the changes into your branch: ```git merge dev```

### 2.2. (Evening) Push Changes:
1. Check what files you changed: ```git status```
2. (Optional, but recommended:) Review your changes: ```git diff```
3. Add files you want to commit: ```git add <FILEPATHES_TO_ADD>```
   Recommended to split by root folders (commit separately, e.g. documentation / data / code)
4. Commit your changes: ```git commit -m "<COMMIT_MESSAGE>"```
   See commit-style.md for detailed info about committing style
5. Push changes to origin repository: ```git push```

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
| Rename variables | Shift + F6 | Shift + F6 |
| Move a code line up/down | Shift + Alt + Arrow up/down | Shift + (Alt or Cmd) + Arrow up/down |
