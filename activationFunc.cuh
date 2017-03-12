/*
 * activationFunc.cuh
 *
 *  Created on: 1 Jul, 2016
 *      Author: chankh
 */

#ifndef af
#define af

struct activate
{
	__host__ __device__
	float operator()(float& x, int& y)
	{
		float k = x;
		switch (y)
		{
		case 0:
			k = x;
			break;
		case 1:
			k = 1.0f / (1.0f + expf(-x));
			break;

		case 2:
			k = tanhf(x);
			break;

		case 3:
			k = atanf(x);
			break;

		case 4:
			k = x / powf((1.0f + fabsf(x)), 2.0f);
			break;

		case 5:
			if (x < 0)
			{
				k = 0;
			}
			else
			{
				k = x;
			}
			break;

		case 6:
			if (x < 0)
			{
				k = expf(x) - 1.0f;
			}
			else
			{
				k = x;
			}
			break;

		case 7:
			k = logf(1.0f + expf(x));
			break;

		case 8:
			k = (sqrtf(x*x + 1.0f) - 1.0f)/2.0f + x;
			break;

		case 9:
			k = sinf(x);
			break;

		case 10:
			if (x < 0.00001f or x > 0.00001f)
			{
				k = 1.0f;
			}
			else
			{
				k = sinf(x) / x;
			}
			break;

		case 11:
			k = expf(-x*x);
			break;
		}
		return k;
	}
};

#endif /* ACTIVATIONFUNC_CUH_ */
