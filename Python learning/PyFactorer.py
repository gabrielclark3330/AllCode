# this program will factor a given number

# things to remember: only have to test prime numbers up to the square root of test 


factorer = 1
looping = False
prime = False

looping = False
factorer = input('Please input the number you would like to factor: ')
print('Thanks for inputing {0}'.format(factorer))

while looping == True:
    if factorer.isalpha:
        looping = False
        factorer = input('Please input the number you would like to factor: ')
        print('Thanks for inputing {0}'.format(factorer))
        break
    else:
        looping = True
        print('there is an error please try again')

print(factorer)

factorer = int(factorer)

if factorer % 2 == 0 and factorer != 2:
    prime = False
elif factorer % 3 == 0 and factorer != 3:
    prime = False
elif factorer % 5 == 0 and factorer != 5:
    prime = False
elif factorer % 7 == 0 and factorer != 7:
    prime = False
else:
    prime = True

print (prime)


