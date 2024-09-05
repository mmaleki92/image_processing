import sqlite3
from datetime import datetime

def calculate_time_spent():
    # Connect to the SQLite database
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()

    # Fetch all records from the logs table
    cursor.execute("SELECT person_id, entry_time, exit_time FROM logs")
    rows = cursor.fetchall()

    # Dictionary to store total time spent by each person
    time_spent = {}

    # Calculate time spent for each person
    for row in rows:
        person_id, entry_time, exit_time = row

        # Ensure both entry and exit times are available
        if entry_time and exit_time:
            # Convert string times to datetime objects
            entry_time_dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
            exit_time_dt = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")

            # Calculate the time difference in seconds
            duration = (exit_time_dt - entry_time_dt).total_seconds()

            # Add the duration to the total time for that person
            if person_id in time_spent:
                time_spent[person_id] += duration
            else:
                time_spent[person_id] = duration

    # Display the total time spent by each person
    for person_id, total_seconds in time_spent.items():
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Person ID: {person_id}, Total Time Spent: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    calculate_time_spent()
