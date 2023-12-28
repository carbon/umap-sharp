using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace UMAP;

internal static class SIMDint
{  
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Uniform(ref float[] data, float a, IRandomValueProvider random)
    {
        float a2 = 2 * a;
        float an = -a;
        random.NextFloats(data);

        TensorPrimitives.Multiply(data, a2, destination: data);
        TensorPrimitives.Add(data, an, destination: data);

        // SIMD.Multiply(ref data, a2);
        // SIMD.Add(ref data, an);
    }
}