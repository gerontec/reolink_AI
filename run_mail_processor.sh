#!/bin/bash
# Email Attachment Processor - Wrapper Script

cd /home/gh/python/reolink_AI

# Aktiviere Virtual Environment
source /home/gh/python/venv_py311/bin/activate

# Führe Mail Processor mit sudo aus (für Maildir-Zugriff)
sudo -E python3 process_mail_attachments.py

exit 0
