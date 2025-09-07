# miniature-dollop Repository
The miniature-dollop repository is currently a minimal repository containing only a basic README.md file. There is no build system, source code, dependencies, or application to run.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Repository State
- Repository contains only README.md with a basic title
- No package.json, build files, or source code exists
- No dependencies or build system configured
- No tests or application to run
- No CI/CD pipelines configured

## Working Effectively
Since this is an empty repository, focus on:
- Use `git status` to check current repository state
- Use `git log --oneline` to view commit history
- Use `ls -la` to see current files in repository root
- Always work from the repository root: `/home/runner/work/miniature-dollop/miniature-dollop`

## Common Operations
- Check repository contents: `ls -la`
- View README: `cat README.md`
- Check Git status: `git status`
- View Git history: `git log --oneline -10`
- Create new files/directories as needed for development

## Validation
- Always verify file operations with `ls -la` after creating/modifying files
- Use `git status` to confirm changes are tracked properly
- No build or test commands exist to validate since there is no build system

## What NOT to Do
- Do not attempt to run build commands - none exist
- Do not attempt to run tests - no test framework configured
- Do not attempt to start applications - no application exists
- Do not look for package.json, Dockerfile, or other build files - they do not exist

## Adding New Development
When adding new functionality to this repository:
- Determine what type of project you want to create (web app, CLI tool, library, etc.)
- Add appropriate package.json, requirements.txt, or equivalent dependency files
- Set up build system and testing framework appropriate for the technology stack
- Update these instructions with specific build, test, and run commands
- Add validation scenarios for the specific application type

## Common Tasks
The following are outputs from frequently run commands. Reference them instead of running bash commands to save time.

### Repository root listing
```
ls -la
total 20
drwxr-xr-x 4 runner docker 4096 Sep  7 18:38 .
drwxr-xr-x 3 runner docker 4096 Sep  7 18:37 ..
drwxr-xr-x 7 runner docker 4096 Sep  7 18:39 .git
drwxr-xr-x 2 runner docker 4096 Sep  7 18:39 .github
-rw-r--r-- 1 runner docker   18 Sep  7 18:37 README.md
```

### README content
```
cat README.md
# miniature-dollop
```

### Git status (typical clean state)
```
git status
On branch copilot/fix-2
Your branch is up to date with 'origin/copilot/fix-2'.

nothing to commit, working tree clean
```

### Git history
```
git log --oneline -10
cdc1fca (HEAD -> copilot/fix-2, origin/copilot/fix-2) Initial plan
c12b84c (grafted) Initial commit
```