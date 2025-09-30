import torch
import random


class CopyPaste(object):
    """
    "We randomly select two images and apply random scale jittering and random
    horizontal flipping on each of them. Then we select a random subset of objects
    from one of the images and paste them onto the other image."
    """
    def __init__(self, keep_prob=0.7, select_prob=0.5, occ_thresh=0.7):
        self.keep_prob = keep_prob
        self.select_prob = select_prob
        self.occ_thresh = occ_thresh

    def get_select_mask(self, mask2):
        return torch.rand(mask2.size(0)) < self.select_prob

    @staticmethod
    def merge_two_images_using_mask(image1, image2, mask):
        """
        "We compute the binary mask ($\alpha$) of pasted objects using ground-truth
        annotations and compute the new image as
        $I_{1} \times \alpha + I_{2} \times (1 - \alpha)$ where $I_{1}$ is
        the pasted image and $I_{2}$ is the main image."
        "To smooth out the edges of the pasted objects we apply a Gaussian filter
        to \alpha similar to “blending” in [13]. Simply composing without any blending has similar performance."
        """
        cat_mask = torch.any(mask, dim=0, keepdim=True).repeat(3, 1, 1)
        # print(cat_mask.dtype)
        return cat_mask * image2 + (~cat_mask) * image1

    @staticmethod
    def get_ltrb_from_mask(mask):
        n_objs = mask.size(0)
        if n_objs == 0:
            return torch.zeros(size=(0, 4), dtype=torch.double)
        else:
            ltrbs = list()
            for batch_idx in range(mask.size(0)):
                nonzero = torch.nonzero(mask[batch_idx])
                # x = nonzero[:, 0]
                # y = nonzero[:, 1]
                y = nonzero[:, 0]
                x = nonzero[:, 1]
                ltrbs.append(
                    torch.tensor([x.min(), y.min(), x.max(), y.max()]).double()
                )
            return torch.stack(ltrbs, dim=0)

    def apply(self, image, annots, idx1, idx2):
        """
        "We remove fully occluded objects and update the masks and bounding boxes
        of partially occluded objects."
        """
        image1 = image[idx1]
        mask1 = annots["masks"][idx1]
        label1 = annots["labels"][idx1]
        ltrb1 = annots["ltrbs"][idx1]
        if idx1 == idx2:
            return image1, mask1, label1, ltrb1

        image2 = image[idx2]
        mask2 = annots["masks"][idx2]
        label2 = annots["labels"][idx2]
        ltrb2 = annots["ltrbs"][idx2]

        select_mask = self.get_select_mask(mask2)
        mask2 = mask2[select_mask]
        label2 = label2[select_mask]
        ltrb2 = ltrb2[select_mask]

        intersec = mask1 & torch.any(mask2, dim=0, keepdim=True)
        new_mask1 = mask1 ^ intersec
        ori_area = torch.sum(mask1, dim=(1, 2))
        rem_area = torch.sum(new_mask1, dim=(1, 2))
        occ_mask = (rem_area / ori_area) > 1 - self.occ_thresh
        # print(idx1, idx2)
        # print(rem_area / ori_area)
        # print(occ_mask)
        new_mask1 = new_mask1[occ_mask]
        mask = torch.cat([new_mask1, mask2], dim=0)
        new_ltrb1 = self.get_ltrb_from_mask(new_mask1)
        ltrb = torch.cat([new_ltrb1, ltrb2], dim=0)
        label = torch.cat([label1[occ_mask], label2], dim=0)

        new_image = self.merge_two_images_using_mask(
            image1=image1, image2=image2, mask=mask2,
        )
        return (new_image, mask, label, ltrb)

    def __call__(self, image, annots):
        batch_size = image.size(0)
        batch_indices1 = list(range(batch_size))
        batch_indices2 = random.sample(range(batch_size), batch_size)
        # print(batch_indices2)

        images = list()
        masks = list()
        labels = list()
        ltrbs = list()
        for idx1, idx2 in zip(batch_indices1, batch_indices2):
            if random.random() < self.keep_prob:
                idx2 = idx2

            image_tensor, mask, label, ltrb = self.apply(
                image, annots, idx1=idx1, idx2=idx2,
            )
            images.append(image_tensor)
            masks.append(mask)
            labels.append(label)
            ltrbs.append(ltrb)
        if not images:
            return
        return (
            torch.stack(images, dim=0),
            {
                "masks": masks,
                "labels": labels,
                "ltrbs": ltrbs,
            },
        )