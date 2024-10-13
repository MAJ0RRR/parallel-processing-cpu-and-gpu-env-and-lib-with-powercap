# Paswordless SSH

Żeby móc uruchamiać aplikacje na wielu maszyncha w labie (des01- des18) trzeba zkofnigurować paswordless SSH między nimi.

1. Wejdź na dowlony desXX.
2. Wygeneruj klucz `ssh-keygen -t rsa -b 4096 -C "sXXXXXX@student.pg.edu.pl"`
3. Skopiuj go na inną maszynę `ssh-copy-id sXXXXXX@desYY` Ten krok należy wykonać tylko raz na dowolnej maszynie i dostęp będzie do wszystkich pozostałych.

Jeśli nadal nie będzie działać to możliwe, że będzie trzeba z maszyny, z której puszczana jest aplikacja chociaż raz połączyć się z każdą z innych maszyn, aby uzupełnić też plik known_hosts, tzn.

1. Będąc na dany desXX robimy `ssh desYY`, jeśli to pierwsze połączenie to spyta nas o fingerprint więc dajemy `yes` i wracamy `exit` i powtarzamy dla wszystkich des, na których checmy puścić aplikację.