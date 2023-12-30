using Carbon.AI.Umap.UnitTests;

namespace Carbon.AI.Umap.Tests;

public sealed class DeterministicRandomGenerator(int seed) : IRandomValueProvider
{
    private readonly Prando _rnd = new(seed);

    public bool IsThreadSafe => false;

    public int Next(int minValue, int maxValue) => _rnd.Next(minValue, maxValue);

    public float NextFloat() => _rnd.NextFloat();

    public void NextFloats(Span<float> buffer)
    {
        for (var i = 0; i < buffer.Length; i++)
        {
            buffer[i] = _rnd.NextFloat();
        }
    }
}