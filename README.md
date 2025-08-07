# Real-time_EEG-Hyperscanning
MNE-LSL based Python software to process two synchronous EEG streams in real-time.

This version of the script expect a synchronization trigger every 3 seconds as a stim stream along with the EEG stream. 
It was designed to be used in conjunction with https://github.com/Brain-Music-Lab/HyperOpenBCI-LSL with two OpenBCI Cyton+Daisy boards connected to a wireless synchronization trigger module (https://github.com/Brain-Music-Lab/OpenBCI-Wireless_Trigger). But modifications to other EEG setups are more than welcomed.
