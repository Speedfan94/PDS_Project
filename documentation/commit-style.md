# Commit Style (Messages)
A commit message should consist of different parts:
* Prefix
* Scope
* Message

And should look like this:
```
git commit -m "<COMMIT_PREFIX>(<COMMIT_SCOPE>): <COMMIT_MESSAGE>"
```

We commit ourselves to write our commit messages to fit the following criterias.

## Prefixes
The following scopes should be used:
* ```feat``` - this commit implements a new feature
* ```fix``` - this commit fixes a bug
* ```wip``` - the code does not work yet (work in progress), commit just saves/uploads code
* ```chore``` - this commit changes anything non-/"not production"-relevant (like adding/changing images, data files, documentation)

## Scopes
Scopes should be used by the root directories.
Files should be committed separately by directory.

The following scopes should be used:
* ```root``` - all files on root level
* ```data``` - data files (like .csv files or images)
* ```documentation``` - documentation files / commitments
* ```nextbike/<DIRECTORY>``` - all module relevant code files, splitted by directory
    (e.g.: "nextbike/io" or "nextbike/model")

## Message Style
The message should fit to the following rules:
* write in english
* write in present tense (don't: "fixed"/"changed"/"added")
* everything is in lowercase
* use zero infinitive form of verbs (don't: "fixes"/"changing"/"added")
* be short and meaningful (don't list all changed files or every code change e.g.)

Quick reminder - The message should complete the following sentence:

"This commit will ..."

## Examples for Good Commit Messages
* chore(root): add project files to gitignore
* chore(documentation): add example section on commit style documentation
* chore(nextbike/io): fix typo in code
* feat(nextbike/io): implement read file method
* fix(nextbike/model): fix import in linear regression model