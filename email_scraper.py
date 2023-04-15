import os
import csv
import email

# Path to folder containing .eml files
folder_path = "fraud_emails"

# Path to CSV file for output
csv_path = "personal_fraud_email.csv"

# Open CSV file for writing
with open(csv_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(["Sender", "Raw", "Fraud"])
    for filename in os.listdir(folder_path):
        if filename.endswith(".eml"):
            with open(os.path.join(folder_path, filename), "r") as file:
                msg = email.message_from_file(file)
                sender = msg["From"]
                raw_message = str(msg)
                
                csv_writer.writerow([sender, raw_message, 1])