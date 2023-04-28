import threading
import asyncio
import websockets




class SocketServer:

    async def handle_client(websocket, path):
        # This function will be called whenever a new client connects to the server.
        # You can use this function to define how to handle incoming messages from the client.
        # For example, you can use a while loop to keep listening for messages until the client disconnects:
        while True:
            message = await websocket.recv()
            print(f"Received message from client: {message}")
            
            # You can also send messages back to the client using the `send()` method:
            response = f"You sent me this message: {message}"
            await websocket.send(response)

    # Define a function to create a new WebSocket server and run it on a separate thread:
    def start_server(host, port):
        asyncio.set_event_loop(asyncio.new_event_loop())
        server = websockets.serve(handle_client, host, port)
        asyncio.get_event_loop().run_until_complete(server)
        asyncio.get_event_loop().run_forever()

# Define the host and port numbers for each server:
servers = [
    ("localhost", 8000),
    ("localhost", 8001),
    ("localhost", 8002)
]

# Create a new thread for each server and start it:
threads = []
for server in servers:
    thread = threading.Thread(target=start_server, args=server)
    thread.daemon = True  # Set the daemon attribute to True
    thread.start()
    threads.append(thread)

# The main thread can exit now
# Daemon threads will automatically terminate
for thread in threads:
    thread.join()