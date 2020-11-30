using System;

namespace PrimeNumberFinder
{
    class displayer //this only works for numbers up to 50 and I think its too slow
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("input the number of values to test");
            string ends = Console.ReadLine();
            int end = int.Parse(ends);
            int counter = 0;
            int[] primes = new int[end];
            for (int i = 0;  i <= end; i++) 
            {
                Console.WriteLine("counter Num" + i);
                if (i % 2 == 0 && i!=2)
                {
                    Console.WriteLine("NotPrime");
                    counter = counter + 1;
                }
                else
                {
                    if (i % 3 == 0 && i!=3)
                    {
                        Console.WriteLine("NotPrime");
                        counter = counter + 1;
                    }
                    else
                    {
                        if (i % 5 == 0 && i!=5)
                        {
                            Console.WriteLine("NotPrime");
                            counter = counter + 1;
                        }
                        else
                        {
                            if (i % 7 == 0 && i != 7)
                            {
                                Console.WriteLine("NotPrime");
                                counter = counter + 1;
                            }
                            else
                            {
                                if (i % 9 == 0 && i != 9)
                                {
                                    Console.WriteLine("NotPrime");
                                    counter = counter + 1;
                                }
                                else
                                {
                                    primes[counter] = i;
                                    counter = counter + 1;
                                }
                            }
                        }
                    }
                }
            }
            foreach (int i in primes)
            {
                if (i != 0)
                {
                    Console.WriteLine("\t" + i);
                }
            }
        }
    }
}
