#!/usr/bin/python
# %% [markdown]
# # # Bi-polar charger testing server part
# # 
#
# %%
import socket
import time 
# %%
if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((socket.gethostname(), 5025))
    s.listen(1)
    # s.settimeout(0.5)

    counter = 0
    increment = True
    while True:
        try:
            # now our endpoint knows about the OTHER endpoint.
            clientsocket, address = s.accept()
            # clientsocket.settimeout(0.5)
            try:
                print(f"Connection from {address} has been established.")
                # clientsocket.send(bytes("Hey there!!!","utf-8"))
                # message = clientsocket.recv(4096)
                # print(message)
                while True:
                    message = clientsocket.recv(4096)
                    print(message)
                    value = str(message[-4])
                    clientsocket.sendall(bytes(f' Confirm: {counter} ' , 'utf-8'))
                    
                    if(increment):
                        counter +=1
                    else:
                        counter -=1
                    if(counter > 10 and increment):
                        increment=False
                    if(counter < -10 and not increment):
                        increment=True
            except:
                print("Error appeared")
                clientsocket.close()
        except:
            s.close()