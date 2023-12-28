using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace UMAP;

internal static class SIMD
{
    private static readonly int _vs1 = Vector<float>.Count;
    private static readonly int _vs2 = 2 * Vector<float>.Count;
    private static readonly int _vs3 = 3 * Vector<float>.Count;
    private static readonly int _vs4 = 4 * Vector<float>.Count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Magnitude(ReadOnlySpan<float> vec)
    {
        return MathF.Sqrt(TensorPrimitives.Dot(vec, vec));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Euclidean(ref float[] lhs, ref float[] rhs)
    {
        float result = 0f;

        var count = lhs.Length;
        var offset = 0;
        Vector<float> diff;
        while (count >= _vs4)
        {
            diff = new Vector<float>(lhs, offset) - new Vector<float>(rhs, offset); result += Vector.Dot(diff, diff);
            diff = new Vector<float>(lhs, offset + _vs1) - new Vector<float>(rhs, offset + _vs1); result += Vector.Dot(diff, diff);
            diff = new Vector<float>(lhs, offset + _vs2) - new Vector<float>(rhs, offset + _vs2); result += Vector.Dot(diff, diff);
            diff = new Vector<float>(lhs, offset + _vs3) - new Vector<float>(rhs, offset + _vs3); result += Vector.Dot(diff, diff);
            if (count == _vs4)
            {
                return result;
            }

            count -= _vs4;
            offset += _vs4;
        }

        if (count >= _vs2)
        {
            diff = new Vector<float>(lhs, offset) - new Vector<float>(rhs, offset); result += Vector.Dot(diff, diff);
            diff = new Vector<float>(lhs, offset + _vs1) - new Vector<float>(rhs, offset + _vs1); result += Vector.Dot(diff, diff);
            if (count == _vs2)
            {
                return result;
            }

            count -= _vs2;
            offset += _vs2;
        }
        if (count >= _vs1)
        {
            diff = new Vector<float>(lhs, offset) - new Vector<float>(rhs, offset); result += Vector.Dot(diff, diff);
            if (count == _vs1)
            {
                return result;
            }

            count -= _vs1;
            offset += _vs1;
        }
        if (count > 0)
        {
            while (count > 0)
            {
                var d = (lhs[offset] - rhs[offset]);
                result += d * d;
                offset++; count--;
            }
        }
        return result;
    }
}