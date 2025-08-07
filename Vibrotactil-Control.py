from pythonosc.udp_client import SimpleUDPClient
import time

# ESP32 IP address and port
esp32_ip1 = "192.168.0.111"  # Replace with your ESP32's IP
esp32_ip2 = "192.168.0.113"
esp32_port = 8000

# Create OSC client
client1 = SimpleUDPClient(esp32_ip1, esp32_port)
client2 = SimpleUDPClient(esp32_ip2, esp32_port)
# Example values (range: 0.0 to 1.0)
pwm1 = 0.0
pwm2 = 0.0

# Send OSC message to /pwm
client1.send_message("/pwm", [pwm1, pwm2])
client2.send_message("/pwm", [pwm1, pwm2])
print(f"Sent /pwm {pwm1}, {pwm2}")

# Optional: loop to send continuously
# while True:
#     client.send_message("/pwm", [pwm1, pwm2])
#     time.sleep(0.1)