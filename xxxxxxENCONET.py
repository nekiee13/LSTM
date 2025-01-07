import os

def clean_srt_file(input_path, output_path):
    try:
        # Open the input file for reading
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # List to store only the content lines
        content_lines = []
        
        # Variable to track if the current line is part of the content
        is_content = False

        for line in lines:
            stripped_line = line.strip()
            # Check if the line is numeric (InputID) or a TimeID
            if stripped_line.isdigit() or '-->' in stripped_line:
                is_content = False
            elif stripped_line:
                # This is a content line
                is_content = True

            if is_content:
                content_lines.append(stripped_line)

        # Write the content lines to the output file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write("\n".join(content_lines))

        print(f"Cleaned content saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define file paths
input_file_path = r"C:\\Users\\user\\pinokio\\api\\whisper-webui.git\\app\\outputs\\Mic-1230014427.srt"
output_file_path = r"C:\\Users\\user\\pinokio\\api\\whisper-webui.git\\app\\outputs\\xENCONET_mic.txt"

# Run the cleaning function
clean_srt_file(input_file_path, output_file_path)
