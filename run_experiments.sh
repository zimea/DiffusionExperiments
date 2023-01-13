rootDir = '.'
relevant_files='config.todo.py'
readarray -d '' todos < <(find . -name "$input" -print0)
