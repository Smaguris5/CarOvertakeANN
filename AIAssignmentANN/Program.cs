using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;

namespace AIAssignmentANN
{
    class Program
    {
        private static readonly List<string> PossibleResults = new List<string> { "TRUE", "FALSE" };
        public static Random Random;

        static void Main(string[] args)
        {
            // Settings
            var filePath = @"OvertakeData.csv";
            var epochs = 5; // Number of training iterations
            var inputNodes = 3;
            var hiddenNodes = 5;
            var outputNodes = 2;
            var learningRate = 0.2;
            Program.RandomSetAsRepeatable(true);

            // Read the data and shuffle it
            var shuffledInputs = GetInputs(filePath);
            var trainDataSetCount = 10000; //Amount of training data

            // Create a network with random weights
            var network = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);

            // Train on a defined data sample
            var trainDataSet = shuffledInputs.Take(trainDataSetCount).ToArray();

            Console.WriteLine($"Training network with {trainDataSet.Length} samples using {epochs} epochs...");
            var s = new Stopwatch();
            s.Start();
            for (var epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var input in trainDataSet)
                {
                    // Extract the 4 data points from the sample
                    var inputList = input.Take(inputNodes).Select(double.Parse).ToArray();

                    // Extract the correct answer and synthesize a 2-node output with 0.01 indicating wrong answer and 0.99 indicating right one.
                    var targets = new[] { 0.01, 0.01 };
                    targets[PossibleResults.IndexOf(input.Last())] = 0.99;

                    // Convert the data to the range 0 - 1.0 (faster to do this outside the epochs loop) 
                    // and train the network.
                    network.Train(NormaliseData(inputList), targets);
                }
            }

            s.Stop();
            Console.WriteLine($"Training complete in{s.ElapsedMilliseconds/1000}s, {s.ElapsedMilliseconds}ms{Environment.NewLine}");

            // Test on the rest of the data samples
            var testDataset = shuffledInputs.Skip(trainDataSetCount).Take(1000).ToArray();

            var scoreCard = new List<bool>();
            foreach (var input in testDataset)
            {
                // The node with the largest value is the answer found
                var result = network.Query(NormaliseData(input.Take(3).Select(double.Parse).ToArray())).ToList();
                var predictedResult = PossibleResults[result.IndexOf(result.Max())];

                // The correct answer is in the final field of the input
                var correctResult = PossibleResults[PossibleResults.IndexOf(input.Last())];

                // Note correctness so we can calculate performance of network
                scoreCard.Add(predictedResult == correctResult);

                var miss = (predictedResult == correctResult) ? "" : "miss";
                Console.WriteLine($"{input[0],4}, {input[1],4}, {input[2],4}, {correctResult,-16}, {predictedResult,-16} {miss}");
            }

            Console.WriteLine(
                $"Performance is {(scoreCard.Count(x => x) / Convert.ToDouble(scoreCard.Count)) * 100} percent.");

            Console.ReadKey();
        }

        private static string[][] GetInputs(string filePath)
        {
            var dataset = File.ReadAllLines(filePath);

            var allInputs = dataset.Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();

            return Program.Shuffle(allInputs);
        }

        public static void RandomSetAsRepeatable(bool repeatable)
        {
            if (repeatable)
                Random = new Random(0);
            else
                Random = new Random();
        }

        public static string[][] Shuffle(string[][] allInputs)
        {
            // The following statement is a shortcut to randomise an array and is equivalent to the longer:
            // return
            //   .Select(i => new { i, rand = random.NextDouble() }) // insert a temporary random key
            //   .OrderBy(x => x.rand) // sort on the random key
            //   .Select(x => x.i)     // remove the key
            //   .ToArray();
            return allInputs.OrderBy(x => Program.Random.NextDouble()).ToArray();
        }
        public static double[][] Shuffle(double[][] allInputs)
        {
            // The following statement is a shortcut to randomise an array and is equivalent to the longer:
            // return
            //   .Select(i => new { i, rand = random.NextDouble() }) // insert a temporary random key
            //   .OrderBy(x => x.rand) // sort on the random key
            //   .Select(x => x.i)     // remove the key
            //   .ToArray();
            return allInputs.OrderBy(x => Program.Random.NextDouble()).ToArray();
        }

        private static double[] NormaliseData(double[] input)
        {
            var maxInitialSeparation = 1000;
            var maxOvertakingSpeed = 100;
            var maxOncomingSpeed = 100;

            var normalised = new[]
            {
                (input[0]/maxInitialSeparation),
                (input[1]/maxOvertakingSpeed),
                (input[2]/maxOncomingSpeed)
            };

            return normalised;
        }
    }
}
