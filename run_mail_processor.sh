#!/bin/bash
# Email Attachment Processor + Chain Trigger

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

source /home/gh/python/venv_py311/bin/activate

python3 process_mail_attachments.py
EXIT_CODE=$?

# Exit 0 = new attachments extracted → run AI chain on web2
if [ ${EXIT_CODE} -eq 0 ]; then
    BASE_PATH="/var/www/web2/$(date +%Y/%m)"
    echo "$(date): New attachments found — triggering chain on ${BASE_PATH}" >> "${SCRIPT_DIR}/logs/mail_chain.log"
    "${SCRIPT_DIR}/run_chain.sh" --base-path "${BASE_PATH}" --limit 50 >> "${SCRIPT_DIR}/logs/mail_chain.log" 2>&1
fi

exit 0
