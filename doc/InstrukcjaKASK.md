
# Konfiguracja VPN 
Na windowsie używam OpenVPN, poniżej moja konfiguracja

```
dev tap
proto tcp-client
persist-key
persist-tun
replay-persist cur-replay-protection.cache
nobind
remote 153.19.55.234 1194
pull
tls-client
cipher AES-256-CBC
ns-cert-type server
tls-auth C:\\Users\\Dell\\Desktop\\vpn2023\\ta_delli50.key 1
ca C:\\Users\\Dell\\Desktop\\vpn2023\\CA_WETI_2020.crt
verb 3
auth-user-pass
tls-cipher "DEFAULT:@SECLEVEL=0"
```

# Łączenie się do labu
Instrukcja oficjalna: https://eti.pg.edu.pl/kask/laboratorium/general-information

1. Rejstrujesz sobie hasło na https://kask.eti.pg.gda.pl/adman/register - uwaga, wiadomości stamtąd lądują mi w spamie na outlooku

### Opcja pierwsza - manualnie
2. Łączysz się do serwera kask.eti.pg.gda.pl - `ssh sXXXXXX@kask.eti.pg.gda.pl` 
3. Z tego serwera możesz łączyć się do maszyn w labie (des01.kask – des18.kask) - tutaj na przykładzie maszyny nr. 5 - `ssh sXXXXXX@des05.kask` 

### Opcja druga - ustawienie w pliku konfiguracyjnym ssh
1. do `~/.ssh/config` albo `C:/Users/Username/.ssh/config` wpisujesz poniższy conifg podmieniając numer indeksu:
```
Host kask
  HostName kask.eti.pg.gda.pl
  User sXXXXXX

Host des01 des02 des03 des04 des05 des06 des07 des08 des09 des10 des11 des12 des13 des14 des15 des16 des17 des18
  HostName %h.kask
  User sXXXXXX
  ProxyJump kask
``` 
2. Łączysz się do danej maszyny w labie `ssh des05` 


4. Korzystasz z serwera, mogą się przydać polecenia w https://eti.pg.edu.pl/kask/laboratorium/rules sekcja "Przydatne polecenia" żeby zobaczyć czy ktoś inny nie korzysta z maszyny 
5. Polecam korzystać z VSCode Remote SSH https://code.visualstudio.com/docs/remote/ssh 