import tkinter as tk
from tkinter import filedialog
import email

# Read file contents given file_path
def read_file(file_path):
    if file_path.endswith(".eml"):
        with open(file_path, 'r') as file:
            # Parse the eml file using the email module
            msg = email.message_from_file(file)
            sender = msg["From"]
            receiver = msg["To"]
            raw_message = str(msg)
            
            # For now, only return receiver 
            return receiver

# Create a new window
window = tk.Tk()
window.title("Dehook Maestro")

# Set the window size
window.geometry("400x300")

# Create a label widget
label = tk.Label(window, text="Email Scraper")
label.pack()

# Create a button widget
button = tk.Button(window, text="Upload File", command=lambda: upload_file())
button.pack()

# Define the upload file function
def upload_file():
    # Ask the user to select a file
    file_path = filedialog.askopenfilename()

    # Read the contents of the file
    file_contents = read_file(file_path)

    # Print the plain text body of the email
    print("Selected file contents:")
    print(file_contents)

# Run the window loop
window.mainloop()
