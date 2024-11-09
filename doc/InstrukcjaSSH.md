# Paswordless SSH

Żeby móc uruchamiać aplikacje na wielu maszyncha w labie (des01- des18) trzeba zkofnigurować paswordless SSH między nimi.

1. Wejdź na dowlony desXX.
2. Wygeneruj klucz `ssh-keygen -t rsa -b 4096 -C "sXXXXXX@student.pg.edu.pl"`
3. Skopiuj go na inną maszynę `ssh-copy-id sXXXXXX@desYY` Ten krok należy wykonać tylko raz na dowolnej maszynie i dostęp będzie do wszystkich pozostałych.
4. Z poziomu maszyny, na której się znajdujesz uruchom skrypt `check_connectiom.sh`. Uzupełni on plik `known_hosts` między wszystkimi obecnie operacyjnymi maszynami w labie.