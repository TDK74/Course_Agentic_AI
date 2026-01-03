import aisuite as ai
import display_functions
import json
import qrcode
import requests

from datetime import datetime
from dotenv import load_dotenv
from IPython.display import Image, display
from qrcode.image.styledpil import StyledPilImage


_ = load_dotenv()

## ------------------------------------------------------ ##
client = ai.Client()

## ------------------------------------------------------ ##
def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

## ------------------------------------------------------ ##
get_current_time()

## ------------------------------------------------------ ##
prompt = "What time is it?"
messages = [{"role" : "user",
             "content" : prompt,
            }]

## ------------------------------------------------------ ##
response = client.chat.completions.create(model = "openai:gpt-4o", messages = messages,
                                          tools = [get_current_time], max_turns = 5)

print(response.choices[0].message.content)

## ------------------------------------------------------ ##
display_functions.pretty_print_chat_completion(response)

## ------------------------------------------------------ ##
tools = [{
        "type" : "function",
        "function" : {
                    "name" : "get_current_time",
                    "description" : "Returns the current time as a string.",
                    "parameters" : {}
                    }
        }]

## ------------------------------------------------------ ##
response = client.chat.completions.create(
                                        model = "openai:gpt-4o",
                                        messages = messages,
                                        tools = tools,
                                        # max_turns = 5
                                        )

## ------------------------------------------------------ ##
print(json.dumps(response.model_dump(), indent = 2, default = str))

## ------------------------------------------------------ ##
response2 = None

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    tool_result = get_current_time()

    messages.append(response.choices[0].message)
    messages.append({"role" : "tool", "tool_call_id" : tool_call.id,
                     "content" : str(tool_result)})

    response2 = client.chat.completions.create(model = "openai:gpt-4o",
                                               messages = messages,
                                               tools = tools, )

    print(response2.choices[0].message.content)

## ------------------------------------------------------ ##
def get_weather_from_ip():
    lat, lon = requests.get('https://ipinfo.io/json').json()['loc'].split(',')

    params = {
            "latitude" : lat,
            "longitude" : lon,
            "current" : "temperature_2m",
            "daily" : "temperature_2m_max,temperature_2m_min",
            "temperature_unit" : "celsius",
            "timezone" : "auto"
            }

    weather_data = requests.get("https://api.open-meteo.com/v1/forecast", params = params).json()
    print("weather data:", weather_data)

    return (
            f"Current: {weather_data['current']['temperature_2m']}°C, "
            f"High: {weather_data['daily']['temperature_2m_max'][0]}°C, "
            f"Low: {weather_data['daily']['temperature_2m_min'][0]}°C"
            )


def write_txt_file(file_path: str, content: str):
    with open(file_path, "w", encoding = "utf-8") as f:
        f.write(content)

    return file_path


def generate_qr_code(data: str, filename: str, image_path: str):
    qr = qrcode.QRCode(error_correction = qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)

    img = qr.make_image(image_factory = StyledPilImage, embedded_image_path = image_path)
    output_file = f"{filename}.png"
    img.save(output_file)

    return f"QR code saved as {output_file} containing: {data[ : 50]}..."

## ------------------------------------------------------ ##
prompt = "Can you get the weather for my location?"

response = client.chat.completions.create(
                                        model = "openai:o4-mini",
                                        messages = [{"role" : "user", "content" : (prompt)
                                                    }],
                                        tools = [
                                                get_current_time,
                                                get_weather_from_ip,
                                                write_txt_file,
                                                generate_qr_code
                                                ],
                                        max_turns = 5
                                        )

display_functions.pretty_print_chat_completion(response)

## ------------------------------------------------------ ##
prompt = ("Can you make a txt note for me called reminders.txt that reminds me to call Daniel "
          "tomorrow at 7PM?")

response = client.chat.completions.create(
                                        model = "openai:o4-mini",
                                        messages = [{"role" : "user", "content" : (prompt)}],
                                        tools = [
                                                get_current_time,
                                                get_weather_from_ip,
                                                write_txt_file,
                                                generate_qr_code
                                                ],
                                        max_turns = 5
                                        )

display_functions.pretty_print_chat_completion(response)

## ------------------------------------------------------ ##
with open('reminders.txt', 'r') as file:
    contents = file.read()
    print(contents)

## ------------------------------------------------------ ##
prompt = ("Can you make a QR code for me using my company's logo that goes to www.deeplearning.ai?"
            " The logo is located at `dl_logo.jpg`. You can call it dl_qr_code.")

response = client.chat.completions.create(
                                        model = "openai:o4-mini",
                                        messages = [{"role" : "user", "content" : (prompt)}],
                                        tools = [
                                                get_current_time,
                                                get_weather_from_ip,
                                                write_txt_file,
                                                generate_qr_code
                                                ],
                                        max_turns = 5
                                        )

display_functions.pretty_print_chat_completion(response)

## ------------------------------------------------------ ##
Image('dl_qr_code.png')

## ------------------------------------------------------ ##
prompt = ("Can you help me create a qr code that goes to www.deeplearning.com from the image "
            "dl_logo.jpg? Also write me a txt note with the current weather please.")

response = client.chat.completions.create(
                                        model = "openai:o4-mini",
                                        messages = [{"role" : "user", "content" : (prompt)}],
                                        tools = [
                                                get_weather_from_ip,
                                                get_current_time,
                                                write_txt_file,
                                                generate_qr_code
                                                ],
                                        max_turns = 10
                                        )

display_functions.pretty_print_chat_completion(response)
