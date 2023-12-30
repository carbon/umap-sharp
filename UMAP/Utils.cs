using System.Linq;
using System.Numerics.Tensors;

namespace Carbon.AI.Umap;

internal static class Utils
{
    /// <summary>
    /// Creates an empty array
    /// </summary>
    public static float[] Empty(int n) => new float[n];

    /// <summary>
    /// Creates an array filled with index values
    /// </summary>
    public static float[] Range(int n) => Enumerable.Range(0, n).Select(i => (float)i).ToArray();

    /// <summary>
    /// Creates an array filled with a specific value
    /// </summary>
    public static float[] Filled(int count, float value)
    {
        var result = new float[count];

        result.AsSpan().Fill(value);

        return result;
    }

    /// <summary>
    /// Returns the mean of an array
    /// </summary>
    public static float Mean(ReadOnlySpan<float> input)
    {
        return TensorPrimitives.Sum(input) / input.Length;
    }

    /// <summary>
    /// Returns the maximum value of an array
    /// </summary>
    public static float Max(ReadOnlySpan<float> input) => TensorPrimitives.Max(input);

    /// <summary>
    /// Generate nSamples many integers from 0 to poolSize such that no integer is selected twice.The duplication constraint is achieved via rejection sampling.
    /// </summary>
    public static int[] RejectionSample(int nSamples, int poolSize, IRandomValueProvider random)
    {
        if(poolSize < nSamples)
        {
            nSamples = poolSize;
        }
        var result = new int[nSamples];
        for (var i = 0; i < nSamples; i++)
        {
            var rejectSample = true;
            while (rejectSample)
            {
                var j = random.Next(0, poolSize);
                var broken = false;
                for (var k = 0; k < i; k++)
                {
                    if (j == result[k])
                    {
                        broken = true;
                        break;
                    }
                }
                if (!broken)
                {
                    rejectSample = false;
                }

                result[i] = j;
            }
        }
        return result;
    }
}