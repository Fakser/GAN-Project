from src.controller import tf

def is_multiple_of_2(x):
    """
    function that returns True if given number x is a multiple of 2
    
    args:
    x: number of type integer that we want to checkk if is a multiple of 2
    
    example:
    >>>is_multiple_of_2(8)
    True
    >>>is_multiple_of_2(15)
    False
    """
    
    if x == 2:
        return True
    elif x and (x%2) == 0 and x!=0:
        return is_multiple_of_2(x/2)
    else:
        return False

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)



"""
wasserstein_gradient_penalty
All losses must be able to accept 1D or 2D Tensors, so as to be compatible with
patchGAN style losses (https://arxiv.org/abs/1611.07004).
To make these losses usable in the TF-GAN framework, please create a tuple
version of the losses with `losses_utils.py`.
"""

def _to_float(tensor):
    return tf.cast(tensor, tf.float32)


# Wasserstein losses from `Wasserstein GAN` (https://arxiv.org/abs/1701.07875).
def wasserstein_generator_loss(discriminator_gen_outputs, weights=1.0, scope=None, loss_collection=tf.compat.v1.GraphKeys.LOSSES, reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS, add_summaries=False):
    """Wasserstein generator loss for GANs.
    See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.
    Args:
        discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `discriminator_gen_outputs`, and must be broadcastable to
        `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
        the same as the corresponding dimension).
        scope: The scope for the operations performed in computing the loss.
        loss_collection: collection to which this loss will be added.
        reduction: A `tf.losses.Reduction` to apply to loss.
        add_summaries: Whether or not to add detailed summaries for the loss.
    Returns:
        A loss Tensor. The shape depends on `reduction`.
    """
    with tf.compat.v1.name_scope(scope, 'generator_wasserstein_loss', (discriminator_gen_outputs, weights)) as scope:
        discriminator_gen_outputs = _to_float(discriminator_gen_outputs)

        loss =  discriminator_gen_outputs
        loss = tf.compat.v1.losses.compute_weighted_loss(loss, weights, scope,
                                                        loss_collection, reduction)

        if add_summaries:
            tf.compat.v1.summary.scalar('generator_wass_loss', loss)

    return loss


def wasserstein_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=tf.compat.v1.GraphKeys.LOSSES,
    reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    """Wasserstein discriminator loss for GANs.
    See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.
    Args:
        discriminator_real_outputs: Discriminator output on real data.
        discriminator_gen_outputs: Discriminator output on generated data. Expected
        to be in the range of (-inf, inf).
        real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `discriminator_real_outputs`, and must be broadcastable to
        `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
        the same as the corresponding dimension).
        generated_weights: Same as `real_weights`, but for
        `discriminator_gen_outputs`.
        scope: The scope for the operations performed in computing the loss.
        loss_collection: collection to which this loss will be added.
        reduction: A `tf.losses.Reduction` to apply to loss.
        add_summaries: Whether or not to add summaries for the loss.
    Returns:
        A loss Tensor. The shape depends on `reduction`.
    """
    with tf.compat.v1.name_scope(scope, 'discriminator_wasserstein_loss', (discriminator_real_outputs, discriminator_gen_outputs, real_weights, generated_weights)) as scope:
        discriminator_real_outputs = _to_float(discriminator_real_outputs)
        discriminator_gen_outputs = _to_float(discriminator_gen_outputs)
        discriminator_real_outputs.shape.assert_is_compatible_with(discriminator_gen_outputs.shape)

        loss_on_generated = tf.compat.v1.losses.compute_weighted_loss(discriminator_gen_outputs, generated_weights, scope, loss_collection=None, reduction=reduction)
        loss_on_real = tf.compat.v1.losses.compute_weighted_loss(discriminator_real_outputs, real_weights, scope, loss_collection=None, reduction=reduction)
        loss = loss_on_generated - loss_on_real
        tf.compat.v1.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.compat.v1.summary.scalar('discriminator_gen_wass_loss',
                                        loss_on_generated)
            tf.compat.v1.summary.scalar('discriminator_real_wass_loss', loss_on_real)
            tf.compat.v1.summary.scalar('discriminator_wass_loss', loss)

    return loss

