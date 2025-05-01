import requests
from analyzer import run_analysis
#our hosting provider
url = "https://team-classifier-api-production.up.railway.app/"

#local host uncomment to test api in your machine
#url = "http://127.0.0.1:8000"




response = requests.get(url + run_analysis("input_video.mp4", "output_dir") + "/analyze")
# Check if the request was successful (status code 200) 

if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    # Print the data
    print(data)
else:
    # Print the error message if the request failed
    print(f"Error: {response.status_code} - {response.text}")
    
print(response)  # => "Our Vitals" 
 # Call the function with your parameters
#############################


