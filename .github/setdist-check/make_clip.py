# 416x240 8-bit 4:2:0, 9 frames: moving gradients, blocks and noise, so the encoder tests many block sizes
import random
random.seed( 7 )
w, h = 416, 240
with open( "clip.yuv", "wb" ) as f:
  for t in range( 9 ):
    y = bytearray( w * h )
    for r in range( h ):
      for c in range( w ):
        v = ( 2 * c + 3 * r + 4 * t ) % 256
        if ( ( c + 3 * t ) // 48 + r // 40 ) % 3 == 0:
          v = ( v * 3 + 50 ) % 256
        y[r * w + c] = max( 0, min( 255, v + random.randint( -6, 6 ) ) )
    uv = bytearray( ( ( c * 7 + r * 5 + 9 * t ) % 256 ) for r in range( h // 2 ) for c in range( w // 2 ) )
    f.write( y ); f.write( uv ); f.write( uv )
