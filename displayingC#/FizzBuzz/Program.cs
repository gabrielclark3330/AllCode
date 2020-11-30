using System;

namespace FizzBuzz
{
    class Program
    {
        static void Main(string[] args)
        {

            Console.WriteLine("Enter the number of digits of FizzBuzz you want to play:");
            string ends = Console.ReadLine();
            int end = Int32.Parse(ends);

            string output = "";

            for (int i = 1; i < end; i++)
            {

            }

            Console.WriteLine(output);

            /*
            Console.WriteLine("Enter the number of digits of FizzBuzz you want to play:");
            string ends = Console.ReadLine();
            int end = Int32.Parse(ends);

            for ( int i = 1; i<= end; i++)
            {
                if (i % 3 == 0 && i % 5 == 0)
                {
                    Console.WriteLine("FizzBuzz");
                }
                else if (i % 3 == 0)
                {
                    Console.WriteLine("Fizz");
                }
                else if (i % 5 == 0)
                {
                    Console.WriteLine("Buzz");
                }
                else
                {
                    Console.WriteLine(i);
                }
            }
            */


        }
    }
}