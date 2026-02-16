#!/usr/bin/env python3
"""
Database Migration: Add annotated_image_path column to cam2_recordings
"""

import pymysql
import sys

db_config = {
    'host': 'localhost',
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

def main():
    try:
        print("Connecting to database...")
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()

        # Check if column exists
        cursor.execute('SHOW COLUMNS FROM cam2_recordings LIKE "annotated_image_path"')
        result = cursor.fetchone()

        if result:
            print('✓ Column annotated_image_path already exists')
        else:
            print('Adding column annotated_image_path...')
            cursor.execute('''
                ALTER TABLE cam2_recordings
                ADD COLUMN annotated_image_path VARCHAR(255) DEFAULT NULL
                AFTER analyzed
            ''')
            conn.commit()
            print('✓ Column added successfully')

        # Verify
        cursor.execute('DESCRIBE cam2_recordings')
        print('\nTable structure:')
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")

        cursor.close()
        conn.close()
        print('\n✓ Migration completed successfully')
        return 0

    except Exception as e:
        print(f'✗ Error: {e}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
