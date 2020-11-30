# Factorial finder in Python without functions

factorializer = input("Please enter the number you would like to take the factorial of:")
checkerx = input("Did you mean to input {0}? enter [Y] for yes and [N] for no: " .format(factorializer))
checker = checkerx.upper()
looping = True
storing = 1

while looping == True:
    if (checker == "N"):
        looping = True
        factorializer = input("Please enter the number you would like to take the factorial of:")
        checkerx = input("Did you mean to input {0}? enter [Y] for yes and [N] for no: " .format(factorializer))
        checker = checkerx.upper()
    elif (checker == "Y"):
        looping = False
        try:
            factorializer = int(factorializer)
            print('You chose the number: {0}'.format(factorializer))
            if factorializer > 0:
                for x in range(1,(factorializer + 1)):
                    storing = x * storing
            else:
                print('The factorial of {0} does not exist'.format(factorializer))
            print('The factorial of {1} is: {0}'.format(storing,factorializer))
            break
        except ValueError:
            print('something went wrong please try again')
    else:
        looping = False
        print('Something doesnt seem right here please try again')

